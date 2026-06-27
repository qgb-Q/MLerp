"""Industrial sampler + honest evaluator for model_lerp. Loads a train.py checkpoint (config+tokenizer stored inside, so
it auto-matches training; robust to torch.compile's _orig_mod. prefix and to ckpts with/without head born/linear-base).
Uses model_lerp's O(1)/token INCREMENTAL decode (prefill prompt once, then step) -- no more full-context recompute.
Three lenses (combine freely):
  generate         python generate.py --prompt "Once upon a time" --max_new 300 --top_p 0.95
  perplexity       python generate.py --ppl ./out_lerp/val_neo.bin            (objective ruler: fluent-in-distribution != strong)
  battery+compare  python generate.py --battery --compare ./out_lerp/ckpt_old.pt   (cross-distribution prompts, side-by-side)
"""
import os,sys,argparse,math,numpy as np,torch,torch.nn.functional as F
from model_lerp import LerpConfig,LerpLM
class ByteTok:
    name='byte'; eot=10
    def encode(self,s): return list(s.encode('utf-8','ignore'))
    def decode(self,ids): return bytes(int(i)&255 for i in ids).decode('utf-8','replace')
class NeoTok:
    name='neo'
    def __init__(self):
        import tiktoken; self.enc=tiktoken.get_encoding('gpt2'); self.eot=self.enc.eot_token
    def encode(self,s): return self.enc.encode_ordinary(s)
    def decode(self,ids): return self.enc.decode([int(i) for i in ids])
def get_tok(name): return ByteTok() if name=='byte' else NeoTok()
BATTERY=[("story","Once upon a time, there was a"),("web","The history of the Roman Empire began"),
    ("technical","To install the package, run the following command:"),("news","Breaking news: scientists announced today that"),
    ("review","I bought this product last week and"),("howto","Here are five tips for improving your")]   # cross-distribution: narrow model fluent only on #1, broad model moderate across all
def load(ckpt,dev):
    ck=torch.load(ckpt,map_location=dev,weights_only=False)
    sd={k.replace('_orig_mod.',''):v for k,v in ck['model'].items()}                                       # strip torch.compile prefix
    cfgd=dict(ck['cfg']); cfgd['head_linear_base']=('head.base.weight' in sd); cfgd['head_born']=any('head.to_re' in k for k in sd)   # match head arch to what's actually in the ckpt
    cfg=LerpConfig(**cfgd); m=LerpLM(cfg).to(dev); m.load_state_dict(sd); m.eval(); tok=get_tok(ck.get('tokenizer','byte'))
    return m,tok,cfg,ck.get('step','?'),ck.get('best_val','?')
@torch.no_grad()
def sample(m,tok,prompt,max_new,temp,top_k,top_p,rep,dev,stop_eot,amp):                                     # O(1)/token incremental decode + temp/top_k/top_p/rep-penalty/stop_eot
    ids=tok.encode(prompt) if prompt else [tok.eot]; x=torch.tensor([ids],dtype=torch.long,device=dev); seen=set(ids); gen=[]
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
        h=m.embed(x); states=[]
        for layer in m.layers: h,st=layer(h,cap=True); states.append(st)                                   # prefill prompt ONCE, O(prompt)
        logits=m.head(m.ln_f(h))[0,-1].float()
        for _ in range(max_new):                                                                           # O(1)/token thereafter
            if rep and rep>1.0:
                for t in seen: logits[t]=logits[t]/rep if logits[t]>0 else logits[t]*rep
            l=logits/max(temp,1e-6)
            if top_k: v,_=torch.topk(l,min(top_k,l.numel())); l[l<v[-1]]=float('-inf')
            if top_p and top_p<1.0:
                s,si=torch.sort(l,descending=True); cum=torch.cumsum(F.softmax(s,-1),-1); rm=cum>top_p; rm[1:]=rm[:-1].clone(); rm[0]=False; l[si[rm]]=float('-inf')
            nxt=torch.multinomial(F.softmax(l,-1),1).item(); gen.append(nxt); seen.add(nxt)
            if stop_eot and nxt==tok.eot: break
            ht=m.embed(torch.tensor([[nxt]],device=dev))                                                    # feed ONLY the new token
            for i,layer in enumerate(m.layers): ht,states[i]=layer.step(ht,states[i])
            logits=m.head(m.ln_f(ht))[0,-1].float()
    return tok.decode(gen)
@torch.no_grad()
def perplexity(m,bin_path,dev,block,n_batches,bs,amp):                                                      # mean CE over evenly-spaced non-overlapping windows -> nats + ppl
    dt=np.uint16 if m.cfg.vocab_size>256 else np.uint8; mm=np.memmap(bin_path,dtype=dt,mode='r'); N=len(mm); step=block*bs
    if N<step+1: return float('nan'),float('nan'),0
    tot=0.0; cnt=0
    for off in np.linspace(0,N-step-1,n_batches).astype(np.int64):
        win=np.stack([mm[off+i*block:off+i*block+block+1] for i in range(bs)])
        x=torch.from_numpy(win[:,:-1].astype(np.int64)).to(dev); y=torch.from_numpy(win[:,1:].astype(np.int64)).to(dev)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp): _,loss=m(x,y)
        tot+=loss.item(); cnt+=1
    L=tot/max(cnt,1); return L,math.exp(L),cnt
def main():
    p=argparse.ArgumentParser()
    p.add_argument('ckpt_pos',nargs='?',default=None,help='checkpoint path (positional; or --ckpt)')
    p.add_argument('--ckpt',default=None); p.add_argument('--compare',default=None,help='second checkpoint -> side-by-side on every output')
    p.add_argument('--prompt',default=''); p.add_argument('--battery',action='store_true',help='cross-distribution prompt battery (exposes narrow vs broad)')
    p.add_argument('--ppl',default=None,help='.bin path for held-out perplexity (the objective metric)')
    p.add_argument('--max_new',type=int,default=300); p.add_argument('--temp',type=float,default=0.8); p.add_argument('--top_k',type=int,default=0)
    p.add_argument('--top_p',type=float,default=0.95); p.add_argument('--rep',type=float,default=1.0); p.add_argument('--num',type=int,default=1)
    p.add_argument('--stop_eot',action='store_true'); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--max_ctx',type=int,default=0,help='(ignored: incremental decode carries full recurrent state -- no context window)')
    p.add_argument('--block',type=int,default=512); p.add_argument('--ppl_batches',type=int,default=40); p.add_argument('--ppl_bs',type=int,default=8)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); a=p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev=a.device; amp=(dev=='cuda')
    ckpt=a.ckpt or a.ckpt_pos
    if not ckpt:
        for c in ['./out_lerp/ckpt_best.pt','./out_lerp/ckpt.pt','out_lerp/ckpt_best.pt','out_lerp/ckpt.pt','ckpt_best.pt','ckpt.pt']:
            if os.path.exists(c): ckpt=c; print(f"[gen] no --ckpt; auto-found {c}"); break
    if not ckpt or not os.path.exists(ckpt):
        print(f"[error] checkpoint not found: {ckpt!r}\n  python generate.py PATH/to/ckpt_best.pt"); sys.exit(1)
    models=[]
    for tag,path in [('A',ckpt)]+([('B',a.compare)] if a.compare else []):
        if not os.path.exists(path): print(f"[error] --compare not found: {path!r}"); sys.exit(1)
        m,tok,cfg,step,bv=load(path,dev); models.append((tag,m,tok))
        print(f"[{tag}] {path} | step={step} val={bv} | {tok.name} V={cfg.vocab_size} params={m.num_params()/1e6:.1f}M L={cfg.n_layers}")
    if a.ppl:
        print(f"\n=== held-out perplexity (objective ruler) on {a.ppl} ===")
        for tag,m,tok in models:
            L,P,c=perplexity(m,a.ppl,dev,a.block,a.ppl_batches,a.ppl_bs,amp)
            print(f"  [{tag}] loss={L:.4f} nats | ppl={P:.2f} | bits/tok={L/math.log(2):.3f}  ({c}x{a.ppl_bs}x{a.block} tok)")
    prompts=BATTERY if a.battery else [(None,a.prompt)]; multi=len(models)>1
    print(f"\n=== generation (incremental O(1)/token; max_new={a.max_new} temp={a.temp} top_k={a.top_k} top_p={a.top_p} rep={a.rep}) ===")
    for label,pr in prompts:
        if label is not None: print(f"\n--- [{label}] {pr!r}")
        for i in range(a.num):
            for tag,m,tok in models:
                out=sample(m,tok,pr,a.max_new,a.temp,a.top_k or 0,a.top_p,a.rep,dev,a.stop_eot,amp)
                pre=f"[{tag}] " if multi else ""; num=f"(#{i+1}) " if a.num>1 else ""
                print(f"{pre}{num}{out if label is not None else pr+out}\n")
if __name__=='__main__': main()
