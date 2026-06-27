"""Diagnose whether model_lerp's stuck-at-~4.0 loss is the FeynmanHead bounded-Born CE FLOOR.
Pure-Born logits live in [cap*tanh(log(eps)/cap), cap] -> max winner-vs-loser gap = cap-floor_logit -> hard CE floor log(1+(V-1)*exp(-gap)) (~3.70 @ V=50257).
PART A (real ckpt): report head config + base.weight health + decompose held-out CE into FULL vs BASE-only (+ Born logit magnitude) -> verdict on whether the Born term is pinning loss.
PART B (synthetic): overfit a tiny token set with born+base / born-only / linear-only heads (using YOUR cap/eps/V) -> shows which heads hit CE~0 and which plateau at the floor.
Run on the GPU box (needs the real ckpt + the .bin for PART A): python diagnose_head.py --ckpt out_lerp/ckpt.pt --bin_dir out_lerp
"""
import argparse,math,os,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from model_lerp import LerpConfig,LerpLM,FeynmanHead
def load(ckpt,dev):                                                          # mirrors generate.py: strip compile prefix + infer head arch from the actual state_dict
    ck=torch.load(ckpt,map_location=dev,weights_only=False)
    sd={k.replace('_orig_mod.',''):v for k,v in ck['model'].items()}
    cfgd=dict(ck['cfg']); cfgd['head_linear_base']=('head.base.weight' in sd); cfgd['head_born']=any('head.to_re' in k for k in sd)
    cfg=LerpConfig(**cfgd); m=LerpLM(cfg).to(dev); m.load_state_dict(sd); m.eval()
    return m,cfg,ck.get('tokenizer','neo')
def floor_value(V,cap,eps):                                                  # theoretical pure-Born CE floor
    lo=cap*math.tanh(math.log(eps)/cap); gap=cap-lo                          # lo=min Born logit(|amp|^2~0); cap=max Born logit -> winner-loser gap
    return math.log(1+(V-1)*math.exp(-gap)),gap,lo
def part_a(m,cfg,dev,bin_dir,tok,block,bs):
    H=m.head; V=cfg.vocab_size; fl,gap,lo=floor_value(V,H.cap,H.eps)
    print("=== PART A: head config + held-out loss decomposition ===")
    print(f"  V={V} d={cfg.d_model} | head_born={H.born_on} head_linear_base={H.base is not None} tie={cfg.tie} | cap={H.cap} eps={H.eps} n_paths={cfg.n_paths}")
    print(f"  Born logit range=[{lo:.3f},{H.cap:.3f}] gap={gap:.3f} -> pure-Born CE floor = {fl:.4f} nats   (your plateau ~4.0 vs this)")
    if H.base is not None:
        bw=H.base.weight; tied=(bw.data_ptr()==m.embed.weight.data_ptr())
        print(f"  base.weight norm={bw.norm().item():.3f} mean|w|={bw.abs().mean().item():.5f} tied_to_embed={tied}")
    else:
        print("  base=None (pure-Born head) -> CE structurally CANNOT drop below the floor above. THIS is the cause; retrain with head_born off + a linear base."); return
    if not bin_dir: print("  (no --bin_dir -> skipping real-data decomposition; run PART B for the floor demo)"); return
    p=os.path.join(bin_dir,f"val_{tok}.bin")
    if not os.path.exists(p): p=os.path.join(bin_dir,f"train_{tok}.bin")
    if not os.path.exists(p): print(f"  (no val_{tok}.bin / train_{tok}.bin in {bin_dir} -> skipping real-data decomposition)"); return
    mm=np.memmap(p,dtype=np.uint16,mode='r'); n=len(mm); ix=np.random.randint(0,n-block-1,size=bs,dtype=np.int64)
    x=torch.tensor(np.stack([mm[i:i+block] for i in ix]).astype(np.int64),device=dev); y=torch.tensor(np.stack([mm[i+1:i+1+block] for i in ix]).astype(np.int64),device=dev)
    with torch.no_grad():
        h=m.embed(x)
        for layer in m.layers: h=layer(h)
        hf=m.ln_f(h); full=m.head(hf); lf=F.cross_entropy(full.reshape(-1,V),y.reshape(-1)).item()
        base=m.head.base(hf); lb=F.cross_entropy(base.reshape(-1,V),y.reshape(-1)).item(); born=full-base
    print(f"  held-out CE: FULL head={lf:.4f} | BASE-only(bypass Born)={lb:.4f}")
    print(f"  logit std: base={base.std().item():.3f}  Born-add={born.std().item():.3f}  (Born std >> base std => Born dominates the softmax)")
    if lb<lf-0.2: print(f"  >>> VERDICT: BASE-only ({lb:.2f}) beats FULL ({lf:.2f}) -> the Born term is HOLDING LOSS UP. FIX: retrain with head_born=False (keeps the unbounded base).")
    elif lf>fl-0.3: print(f"  >>> VERDICT: FULL sits at the Born floor ({fl:.2f}) and base can't sharpen past it -> base is being dominated / can't grow (likely the ADDED +-{H.cap} Born term, worsened by tie). FIX: head_born=False, or untie the base.")
    else: print(f"  >>> VERDICT: FULL ({lf:.2f}) is below the floor -> the head is NOT the bottleneck. Look upstream (data/targets/a layer killing gradient). Run PART B to confirm the head can reach ~0.")
def part_b(dev,V,d,cap,eps,n_paths,steps,B,T,plot):
    print("\n=== PART B: head expressivity probe (overfit {} random tokens; can each head reach CE~0?) ===".format(B*T))
    tgt=torch.randint(0,V,(B,T),device=dev); fl,_,_=floor_value(V,cap,eps); curves={}
    for name,kw in [('born+base (your config)',dict(linear_base=True,born=True)),('born-only (no base)',dict(linear_base=False,born=True)),('linear-only (no born)',dict(linear_base=False,born=False))]:
        torch.manual_seed(0); h=nn.Parameter(0.02*torch.randn(B,T,d,device=dev)); head=FeynmanHead(d,V,n_paths,eps,cap,**kw).to(dev)
        opt=torch.optim.Adam(list(head.parameters())+[h],lr=3e-3); hist=[]
        for s in range(steps):
            opt.zero_grad(); loss=F.cross_entropy(head(h).reshape(-1,V),tgt.reshape(-1)); loss.backward(); opt.step()
            if s%max(1,steps//40)==0 or s==steps-1: hist.append((s,loss.item()))
        curves[name]=hist; print(f"  {name:<26} final CE={hist[-1][1]:.4f}")
    print(f"  pure-Born theoretical floor @ V={V} cap={cap} eps={eps} = {fl:.4f}")
    print("  >>> if born+base ALSO plateaus near the floor while linear-only -> ~0, the base CANNOT rescue the Born floor in your config (retrain head_born off).")
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        plt.figure(figsize=(8,5))
        for name,hist in curves.items(): plt.plot([a for a,_ in hist],[b for _,b in hist],'o-',ms=3,label=name)
        plt.axhline(fl,color='red',ls='--',lw=1,label=f'Born floor={fl:.2f}'); plt.xlabel('overfit step'); plt.ylabel('CE (nats)')
        plt.title(f'FeynmanHead expressivity probe (V={V}, cap={cap}, eps={eps})'); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(plot,dpi=120); print(f"  [plot] -> {plot}")
    except Exception as e: print(f"  [warn] plot skipped: {e}")
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ckpt',required=True); p.add_argument('--bin_dir',default=None,help='dir with val_<tok>.bin / train_<tok>.bin for the real-data decomposition')
    p.add_argument('--block',type=int,default=512); p.add_argument('--bs',type=int,default=8)
    p.add_argument('--probe_steps',type=int,default=400); p.add_argument('--probe_dim',type=int,default=256,help='hidden dim for PART B (small=fast; floor is dim-independent)')
    p.add_argument('--probe_bt',default='4,32',help='B,T tokens to overfit in PART B'); p.add_argument('--no_probe',action='store_true')
    p.add_argument('--plot_out',default='head_probe.png'); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args(); dev=a.device; m,cfg,tok=load(a.ckpt,dev)
    part_a(m,cfg,dev,a.bin_dir,tok,a.block,a.bs)
    if not a.no_probe:
        B,T=[int(x) for x in a.probe_bt.split(',')]; part_b(dev,cfg.vocab_size,a.probe_dim,m.head.cap,m.head.eps,cfg.n_paths,a.probe_steps,B,T,a.plot_out)
if __name__=='__main__': main()
