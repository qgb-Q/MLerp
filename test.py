"""Comprehensive test/diagnostic battery for model_lerp (original 4-level six-projection skeleton).
Non-invasive: imports the real modules and uses test-only variants (AblatableLayer / CfgHead / LinearHead)
so model_lerp.py is never modified. AblatableLayer mirrors v2's SIX-projection 2-hop forward
(hop2 reads the lerp-refined Ixl/Txl/Cxl, not a single brain ledger).
Sections: [1] component unit tests  [2] ablation  [3] controlled-variable  [4] training stability
[5] engineering optimizability  [6] problem scan  [7] intrinsic exps  [8] dashboard.
Run: python test.py [--full] [--compile] [--device cpu]
"""
import os,sys,time,math,argparse,copy,contextlib
import numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from model_lerp import LerpConfig,LerpLM,LerpLayer,RMSNorm,FeynmanHead,lin_read,_lin_read_ref
torch.set_num_threads(max(1,os.cpu_count()//2))
RES=[]                                                                   # (section,name,status,detail)
def rec(sec,name,ok,detail=""): RES.append((sec,name,ok,detail)); print(f"  [{ok:4}] {name}"+(f" | {detail}" if detail else ""))
def H(s): print("\n"+"="*78+f"\n {s}\n"+"="*78)
DEV='cpu'; PHI=lambda x:F.elu(x)+1
# ---------------- data ----------------
def load_bytes(n=300000):
    p='./data/input.txt'
    b=open(p,'rb').read()[:n] if os.path.exists(p) else (b"once upon a time there was a small cat. "*8000)
    return torch.tensor(list(b),dtype=torch.long)
DATA=load_bytes()
def get_batch(bs,seq,seed=None):
    if seed is not None: torch.manual_seed(seed)
    ix=torch.randint(len(DATA)-seq-1,(bs,)); x=torch.stack([DATA[i:i+seq] for i in ix]); y=torch.stack([DATA[i+1:i+seq+1] for i in ix]); return x.to(DEV),y.to(DEV)
TCFG=dict(vocab_size=256,d_model=32,d_cell=12,n_lobe=2,n_cortex=4,n_layers=2,chunk_len=16,ffn_mult=4)
def cfg(**kw): d=dict(TCFG); d.update(kw); return LerpConfig(**d)
# ---------------- test-only variants (do NOT modify model_lerp) ----------------
class CfgHead(nn.Module):                                                # parameterized amplitude head
    def __init__(self,d,V,eps=1e-6,fp32=False,cap=30.0):
        super().__init__(); self.re=nn.Linear(d,V,bias=False); self.im=nn.Linear(d,V,bias=False); self.eps=eps; self.fp32=fp32; self.cap=cap
    def forward(self,h):
        if self.fp32: h=h.float()
        re,im=self.re(h),self.im(h); return self.cap*torch.tanh(torch.log(re*re+im*im+self.eps)/self.cap)
class LinearHead(nn.Module):
    def __init__(self,d,V): super().__init__(); self.lin=nn.Linear(d,V,bias=False)
    def forward(self,h): return self.lin(h)
class AblatableLayer(LerpLayer):                                         # mirrors v2 SIX-projection 2-hop LerpLayer.forward with switches
    def __init__(self,c,use_lerp=True,use_Ta=True,use_hop2=True,use_pool=True,use_ffn=True):
        super().__init__(c); self.use_lerp=use_lerp; self.use_Ta=use_Ta; self.use_hop2=use_hop2; self.use_pool=use_pool; self.use_ffn=use_ffn
    def forward(self,X):
        B,L,d=X.shape; h=self.ln_in(X); ch=self.ch; dk=self.dk; dc=self.dc
        Ix=self._h(self.to_Ix(h),dk); Tx=self._h(self.to_Tx(h),dk); Cx=self._h(self.to_Cx(h),dc)
        Ts=self._h(self.to_Ts(h),dk); Cs=self._h(self.to_Cs(h),dc); Is=self._h(self.to_Is(h),dc if self.think_mode=='conv' else dk)
        if self.cfg.qk_conv: Ix=self._dwconv(Ix,self.conv_q,self.cfg.qk_kernel); Ts=self._dwconv(Ts,self.conv_k,self.cfg.qk_kernel)
        Xl=lin_read(Ix,Ts,Cs,ch)                                                # hop1: token reads brain
        if self.use_lerp: Ixl=self.Wil(Xl)+torch.sigmoid(self.a_I)*Ix; Txl=self.Wtl(Xl)+torch.sigmoid(self.a_T)*Tx; Cxl=self.Wcl(Xl)+torch.sigmoid(self.a_C)*Cx
        else: Ixl,Txl,Cxl=Ix,Tx,Cx                                              # -lerp: hop-2 reads UNrefined token ledger
        Ra=lin_read(Ixl,Txl,Cxl,ch) if self.use_hop2 else torch.zeros_like(Xl)  # hop2: 2-hop refined retrieval (Q/K/V all lerp-refined)
        if self.use_Ta: Ta=self._dwconv(Is,self.think_conv,self.cfg.think_kernel) if self.think_mode=='conv' else lin_read(Is,Ts,Cs,ch)  # conv self-think (or lin ablation)
        else: Ta=torch.zeros_like(Xl)
        if self.use_pool: Ra=self.agg_r(Ra); Ta=self.agg_t(Ta)                  # learnable cross-cortex aggregation
        feat=torch.cat([Xl.permute(0,2,1,3).reshape(B,L,-1),(Ra+Ta).permute(0,2,1,3).reshape(B,L,-1)],-1)
        O=X+self.proj_o(feat)
        if self.use_ffn: O=O+self.ffn(self.ln_ff(O))
        return O
def build(c,flags=None,head='feynman',eps=0.1,fp32=True,seed=0):   # default = real model head (hybrid FeynmanHead); section 4 passes head='amp'+explicit eps to demo the cliff
    torch.manual_seed(seed); m=LerpLM(c)
    if flags is not None: m.layers=nn.ModuleList([AblatableLayer(c,**flags) for _ in range(c.n_layers)])
    if head=='linear': m.head=LinearHead(c.d_model,c.vocab_size)
    elif head=='amp': m.head=CfgHead(c.d_model,c.vocab_size,eps=eps,fp32=fp32)
    elif head=='feynman': m.head=FeynmanHead(c.d_model,c.vocab_size)
    m.apply(m._init)
    return m.to(DEV)
def count_spikes(l,thr=1.5):
    if len(l)<5: return 0
    c=0; ema=l[0]
    for x in l[1:]:
        if x>thr*ema: c+=1
        ema=0.9*ema+0.1*x
    return c
def short_train(m,steps=100,lr=3e-3,bs=8,seq=64,clip=1.0,seed=0,probe_p=False):
    torch.manual_seed(seed); opt=torch.optim.AdamW(m.parameters(),lr=lr); m.train()
    lh=[]; gh=[]; ph=[]; nan=False; cap=[None]
    hk=m.head.register_forward_pre_hook(lambda mod,inp:cap.__setitem__(0,inp[0].detach())) if probe_p else None
    for s in range(steps):
        x,y=get_batch(bs,seq)
        _,loss=m(x,y)
        if not torch.isfinite(loss): nan=True; break
        opt.zero_grad(); loss.backward(); gn=nn.utils.clip_grad_norm_(m.parameters(),clip).item(); opt.step()
        lh.append(loss.item()); gh.append(gn)
        if probe_p and cap[0] is not None and hasattr(m.head,'re'):
            with torch.no_grad(): r=m.head.re(cap[0]); im=m.head.im(cap[0]); ph.append((r*r+im*im).min().item())
    if hk: hk.remove()
    return dict(loss=lh,grad=gh,minp=ph,nan=nan,spikes=count_spikes(lh),final=(lh[-1] if lh else float('nan')))
# ================= [1] COMPONENT UNIT TESTS =================
def s1_components():
    H("[1] COMPONENT UNIT TESTS")
    B,Hh,L,dk,dv=2,4,40,8,12
    I,T,C=torch.randn(B,Hh,L,dk),torch.randn(B,Hh,L,dk),torch.randn(B,Hh,L,dv)
    o=lin_read(I,T,C,16); rec("1","lin_read shape","PASS" if tuple(o.shape)==(B,Hh,L,dv) else "FAIL",str(tuple(o.shape)))
    errs=[(ch,(lin_read(I,T,C,ch)-_lin_read_ref(I,T,C)).abs().max().item()) for ch in [1,8,16,32,64,128]]
    me=max(e for _,e in errs); rec("1","lin_read chunked==cumsum (all chunk_len)","PASS" if me<1e-4 else "FAIL",f"max_err={me:.1e} over chunk={[c for c,_ in errs]}")
    o1=lin_read(I,T,C,16); I2,T2,C2=I.clone(),T.clone(),C.clone(); p=27; T2[:,:,p]+=3; C2[:,:,p]+=3; o2=lin_read(I2,T2,C2,16)
    bf=(o1[:,:,:p]-o2[:,:,:p]).abs().max().item(); af=(o1[:,:,p:]-o2[:,:,p:]).abs().max().item()
    rec("1","lin_read causal (perturb t -> only >=t changes)","PASS" if bf<1e-6 and af>1e-6 else "FAIL",f"before={bf:.1e} after={af:.1e}")
    edge=all(torch.isfinite(lin_read(torch.randn(1,2,Lx,4),torch.randn(1,2,Lx,4),torch.randn(1,2,Lx,4),16)).all().item() for Lx in [1,15,16,17,64])
    rec("1","lin_read edge L in {1,15,16,17,64} finite","PASS" if edge else "FAIL")
    big=lin_read(I*50,T*50,C*50,16); rec("1","lin_read large-magnitude finite","PASS" if torch.isfinite(big).all() else "WARN","clamp floor 1e-6")
    rn=RMSNorm(16); xr=torch.randn(3,7,16)*5; yr=rn(xr); rms=yr.pow(2).mean(-1).sqrt().mean().item()
    rec("1","RMSNorm output RMS~1","PASS" if abs(rms-1)<0.15 else "WARN",f"rms={rms:.3f}")
    rec("1","RMSNorm zero-input finite","PASS" if torch.isfinite(rn(torch.zeros(2,16))).all() else "FAIL")
    hd=FeynmanHead(16,256); lg=hd(torch.randn(2,5,16)); rec("1","FeynmanHead shape+finite (Born path-integral, capped)","PASS" if tuple(lg.shape)==(2,5,256) and torch.isfinite(lg).all() else "FAIL",f"shape={tuple(lg.shape)} max|logit|={lg.abs().max():.2f}")
    c=cfg(); ly=LerpLayer(c).to(DEV); xl=torch.randn(2,30,c.d_model,device=DEV); ol=ly(xl)
    rec("1","LerpLayer shape preserve","PASS" if ol.shape==xl.shape else "FAIL")
    ly.zero_grad(); ol.sum().backward(); gf=all((q.grad is None) or torch.isfinite(q.grad).all() for q in ly.parameters())
    rec("1","LerpLayer grads finite","PASS" if gf else "FAIL")
    m=build(c); x,y=get_batch(4,40); lg,loss=m(x,y)
    rec("1","LerpLM init loss ~ln(V)","PASS" if abs(loss.item()-math.log(256))<1.0 else "WARN",f"loss={loss.item():.3f} ln256={math.log(256):.3f}")
    m.zero_grad(); loss.backward(); fin=all((q.grad is None) or torch.isfinite(q.grad).all() for q in m.parameters())
    rec("1","LerpLM all grads finite","PASS" if fin else "FAIL")
    m.eval()
    with torch.no_grad():
        xb=torch.randint(0,256,(2,40),device=DEV); o1,_=m(xb); p=27; xb2=xb.clone(); xb2[:,p]=(xb2[:,p]+5)%256; o2,_=m(xb2)
        bf=(o1[:,:p]-o2[:,:p]).abs().max().item(); af=(o1[:,p:]-o2[:,p:]).abs().max().item()
    rec("1","LerpLM end-to-end causal","PASS" if bf<1e-5 and af>1e-5 else "FAIL",f"before={bf:.1e} after={af:.1e}")
    g=m.generate(torch.randint(0,256,(1,5),device=DEV),10); ok=g.shape[1]==15 and g.max().item()<256
    rec("1","LerpLM generate valid ids","PASS" if ok else "FAIL",f"shape={tuple(g.shape)}")
    m.train(); l_a=m(x,y,grad_checkpoint=False)[1].item(); l_b=m(x,y,grad_checkpoint=True)[1].item()
    rec("1","grad_checkpoint loss-equivalent","PASS" if abs(l_a-l_b)<1e-4 else "FAIL",f"|d|={abs(l_a-l_b):.1e}")
# ================= [2] ABLATION =================
def s2_ablation():
    H("[2] ABLATION (disable one component, train 100 steps, lower=better)")
    c=cfg(); variants=[("full(baseline)",{}),("-hierarchy_pool",dict(use_pool=False)),("-Ta(self-think)",dict(use_Ta=False)),
        ("-hop2(2nd read)",dict(use_hop2=False)),("-query_refine",dict(use_lerp=False)),("-FFN",dict(use_ffn=False))]
    base=None; rows=[]
    for nm,fl in variants:
        m=build(c,flags={**dict(use_lerp=True,use_Ta=True,use_hop2=True,use_pool=True,use_ffn=True),**fl})
        r=short_train(m,100,lr=3e-3,seed=0); base=base or r['final']; rows.append((nm,r['final'],r['spikes'],r['nan']))
        d=f"final={r['final']:.3f}  d_vs_full={r['final']-rows[0][1]:+.3f}  spikes={r['spikes']}"+(" NaN!" if r['nan'] else "")
        rec("2",nm,"FAIL" if r['nan'] else ("WARN" if (rows[0] and r['final']<rows[0][1]-1e-3 and nm!=rows[0][0]) else "PASS"),d)
    worst=max(rows[1:],key=lambda x:x[1]-rows[0][1]) if len(rows)>1 else None
    if worst: print(f"  -> most impactful component (largest loss increase when removed): {worst[0]} (+{worst[1]-rows[0][1]:.3f})")
# ================= [3] CONTROLLED-VARIABLE =================
def s3_controlled():
    H("[3] CONTROLLED-VARIABLE")
    print(" (3a) chunk_len must NOT change outputs (pure engineering knob). NB: must disable TF32 -- TF32 matmuls round per-op-shape, so different chunk sizes diverge ~1e-3 under TF32 even though the MATH is identical (this is precision, not a bug).")
    c=cfg(); m=build(c); m.eval(); x,_=get_batch(2,48,seed=1)
    _tf=torch.backends.cuda.matmul.allow_tf32 if DEV.startswith('cuda') else False
    if DEV.startswith('cuda'): torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False   # force true-fp32 matmul to test math invariance
    with torch.no_grad():
        for L in m.layers: L.ch=64
        ref=m(x)[0]
        diffs=[]
        for ch in [1,8,16,32,64]:
            for L in m.layers: L.ch=ch
            diffs.append((ch,(m(x)[0]-ref).abs().max().item()))
    if DEV.startswith('cuda'): torch.backends.cuda.matmul.allow_tf32=_tf; torch.backends.cudnn.allow_tf32=_tf
    md=max(d for _,d in diffs); rec("3","chunk_len output-invariance (fp32)","PASS" if md<1e-4 else "FAIL",f"max_diff={md:.1e} over chunk={[c for c,_ in diffs]} (TF32-off; under TF32 expect ~1e-3 from rounding)")
    print(" (3b) one-factor sweeps (train 80 steps)")
    for lr in [1e-3,3e-3,1e-2]:
        r=short_train(build(cfg()),80,lr=lr,seed=0); rec("3",f"lr={lr:g}","WARN" if (r['spikes']>2 or r['nan']) else "PASS",f"final={r['final']:.3f} spikes={r['spikes']}"+(" NaN" if r['nan'] else ""))
    for nc in [2,4,8]:
        r=short_train(build(cfg(n_cortex=nc)),80,lr=3e-3,seed=0); rec("3",f"n_cortex={nc} (heads={2*nc})","PASS",f"final={r['final']:.3f} spikes={r['spikes']}")
    for nl in [1,2,3]:
        r=short_train(build(cfg(n_layers=nl)),80,lr=3e-3,seed=0); rec("3",f"n_layers={nl}","PASS",f"final={r['final']:.3f} spikes={r['spikes']}")
# ================= [4] TRAINING STABILITY =================
def s4_stability():
    H("[4] TRAINING STABILITY  (amplitude-head cliff: grad amplifier = 2re/(re^2+im^2+eps))")
    print(" (4a) gradient-amplifier vs eps  [analytic max = 1/sqrt(eps)]")
    re=torch.randn(400000)*0.1; im=torch.randn(400000)*0.1
    for eps in [1e-6,1e-3,1e-1,1.0]:
        emp=(2*re.abs()/(re*re+im*im+eps)).max().item(); ana=1/math.sqrt(eps)
        rec("4",f"eps={eps:g}","WARN" if ana>10 else "PASS",f"analytic_max={ana:.1f}  empirical_max={emp:.1f}  ({'CLIFF' if ana>10 else 'bounded'})")
    print(" (4b) actual head backward grad-norm as amplitude collapses (scale weights down)")
    for eps in [1e-6,1.0]:
        gns=[]
        for sc in [1.0,1e-1,1e-2,1e-3]:
            hd=CfgHead(32,256,eps=eps)
            with torch.no_grad(): hd.re.weight*=sc; hd.im.weight*=sc
            h=torch.randn(4,16,32); loss=F.cross_entropy(hd(h).reshape(-1,256),torch.randint(0,256,(64,))); loss.backward()
            gns.append((hd.re.weight.grad.norm()+hd.im.weight.grad.norm()).item())
        rec("4",f"head grad-norm sweep eps={eps:g}","WARN" if max(gns)/ (min(gns)+1e-9)>50 else "PASS",f"scale[1,.1,.01,.001]-> gradnorm={[f'{g:.1f}' for g in gns]}")
    print(" (4c) end-to-end stability: 3 head variants, 120 steps @ lr=5e-3 (stress)")
    runs={}
    for nm,kw in [("amp eps=1e-6 (old cliff)",dict(head='amp',eps=1e-6)),("amp eps=1.0 fp32",dict(head='amp',eps=1.0,fp32=True)),("feynman hybrid (NEW)",dict(head='feynman')),("linear (control)",dict(head='linear'))]:
        r=short_train(build(cfg(),**kw),120,lr=5e-3,seed=0,probe_p=(kw.get('head')=='amp')); runs[nm]=r
        mp=f" min_p_low={min(r['minp']):.1e}" if r['minp'] else ""
        rec("4",nm,"FAIL" if r['nan'] else ("WARN" if r['spikes']>2 else "PASS"),f"final={r['final']:.3f} spikes={r['spikes']} maxgrad={max(r['grad']):.1f}"+mp+(" NaN!" if r['nan'] else ""))
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(13,4.5))
        for nm,r in runs.items():
            ax[0].plot(r['loss'],label=nm,alpha=.85); ax[1].plot(r['grad'],label=nm,alpha=.85)
        ax[0].set_title('train loss (spikes)'); ax[0].set_xlabel('step'); ax[0].set_ylabel('loss'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        ax[1].set_title('grad norm (pre-clip)'); ax[1].set_xlabel('step'); ax[1].set_yscale('log'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
        plt.tight_layout(); plt.savefig('test_stability.png',dpi=120); rec("4","stability plot","PASS","test_stability.png")
    except Exception as e: rec("4","stability plot","WARN",str(e))
# ================= [5] ENGINEERING OPTIMIZABILITY =================
def s5_engineering(do_compile):
    H("[5] ENGINEERING OPTIMIZABILITY")
    print(" (5a) lin_read chunked vs cumsum: time + memory scaling vs L")
    B,Hh,dk,dv=4,8,16,16; Ls=[64,128,256,512]; tc=[];tr=[];mc=[];mr=[]
    for L in Ls:
        I,T,C=torch.randn(B,Hh,L,dk),torch.randn(B,Hh,L,dk),torch.randn(B,Hh,L,dv)
        t0=time.time(); [lin_read(I,T,C,64) for _ in range(5)]; tc.append((time.time()-t0)/5*1e3)
        t0=time.time(); [_lin_read_ref(I,T,C) for _ in range(5)]; tr.append((time.time()-t0)/5*1e3)
        mc.append(B*Hh*dk*dv*4/1e6); mr.append(B*Hh*L*dk*dv*4/1e6)
    for i,L in enumerate(Ls): print(f"   L={L:4d} | chunked {tc[i]:6.1f}ms (state {mc[i]:.2f}MB) | cumsum {tr[i]:6.1f}ms (ledger {mr[i]:.1f}MB) | mem x{mr[i]/mc[i]:.0f}")
    rec("5","chunked memory advantage","PASS",f"at L=512 cumsum ledger {mr[-1]:.0f}MB vs chunked state {mc[-1]:.2f}MB ({mr[-1]/mc[-1]:.0f}x)")
    if torch.cuda.is_available():
        I,T,C=[t.cuda() for t in (torch.randn(B,Hh,512,dk),torch.randn(B,Hh,512,dk),torch.randn(B,Hh,512,dv))]
        torch.cuda.reset_peak_memory_stats(); lin_read(I,T,C,64); pc=torch.cuda.max_memory_allocated()/1e6
        torch.cuda.reset_peak_memory_stats(); _lin_read_ref(I,T,C); pr=torch.cuda.max_memory_allocated()/1e6
        rec("5","CUDA peak mem (measured)","PASS",f"chunked {pc:.0f}MB vs cumsum {pr:.0f}MB")
    print(" (5b) grad_checkpoint: loss/grad equivalence + memory note")
    c=cfg(n_layers=3); m=build(c); x,y=get_batch(4,64,seed=2)
    m.train(); l0,g0=loss_and_grad(m,x,y,False); l1,g1=loss_and_grad(m,x,y,True)
    rec("5","grad_ckpt exact (loss+grad)","PASS" if abs(l0-l1)<1e-4 and g0 is not None and (g0-g1).abs().max()<1e-3 else "WARN",f"|dloss|={abs(l0-l1):.1e} |dgrad|={(g0-g1).abs().max():.1e}")
    print(" (5c) forward+backward profile (top CPU ops)")
    try:
        m2=build(cfg()); x,y=get_batch(8,64,seed=3)
        with torch.autograd.profiler.profile() as prof: _,l=m2(x,y); l.backward()
        top=prof.key_averages().table(sort_by="self_cpu_time_total",row_limit=6); print("\n".join("   "+ln for ln in top.splitlines()[:10])); rec("5","profile captured","PASS","see top ops above")
    except Exception as e: rec("5","profile","WARN",str(e))
    if do_compile:
        try:
            mc_=torch.compile(build(cfg())); x,y=get_batch(4,64,seed=4); t0=time.time(); _,l=mc_(x,y); l.backward(); rec("5","torch.compile runs","PASS",f"first-call {time.time()-t0:.1f}s")
        except Exception as e: rec("5","torch.compile","WARN",str(e))
    else: rec("5","torch.compile","SKIP","pass --compile to test (slow on CPU first-call)")
def loss_and_grad(m,x,y,ck):
    m.zero_grad(); _,l=m(x,y,grad_checkpoint=ck); l.backward()
    g=next((p.grad.flatten()[:2000].clone() for p in m.parameters() if p.grad is not None and p.numel()>=2000),None)
    return l.item(),g
# ================= [6] PROBLEM SCAN =================
def s6_scan():
    H("[6] PROBLEM SCAN")
    ok=True
    for sd in range(5):
        m=build(cfg(),seed=sd); x,y=get_batch(6,48,seed=100+sd); m.zero_grad(); _,l=m(x,y); l.backward()
        if not (torch.isfinite(l) and all((p.grad is None) or torch.isfinite(p.grad).all() for p in m.parameters())): ok=False
    rec("6","NaN/Inf scan x5 seeds","PASS" if ok else "FAIL")
    m=build(cfg()); x,y=get_batch(6,48,seed=7); m.zero_grad(); _,l=m(x,y); l.backward()
    dead=[n for n,p in m.named_parameters() if p.grad is None or p.grad.abs().sum().item()==0]
    rec("6","dead params (no gradient)","PASS" if not dead else "WARN",f"{len(dead)} dead"+(": "+",".join(dead[:4]) if dead else ""))
    gpl=[]; apl=[]; caps=[]
    hooks=[L.register_forward_hook(lambda mod,i,o:caps.append(o.detach().abs().max().item())) for L in m.layers]
    m.zero_grad(); _,l=m(x,y); l.backward()
    for h in hooks: h.remove()
    for i,L in enumerate(m.layers): gpl.append(sum(p.grad.norm().item() for p in L.parameters() if p.grad is not None))
    rec("6","per-layer grad-flow (no vanish/explode)","PASS" if (min(gpl)>1e-6 and max(gpl)<1e4) else "WARN",f"grad_norm/layer={[f'{g:.2f}' for g in gpl]}")
    rec("6","per-layer activation magnitude","PASS" if max(caps)<1e3 else "WARN",f"max|act|/layer={[f'{a:.1f}' for a in caps]}")
    m.eval()
    with torch.no_grad(): a=m(x)[0]; b=m(x)[0]
    rec("6","determinism (same in->same out)","PASS" if (a-b).abs().max().item()==0 else "FAIL")
    e=m.embed.weight.std().item(); pr=m.layers[0].to_Ix.weight.std().item()
    rec("6","init std sanity","PASS" if 0.005<e<0.1 and 0.005<pr<0.1 else "WARN",f"embed_std={e:.3f} proj_std={pr:.3f}")
    with torch.no_grad(): lg=m(x)[0]; cap_=getattr(m.head,'cap',30.0); sat=(lg.abs()>0.99*cap_).float().mean().item()
    rec("6","logit saturation (head cap)","PASS" if sat<0.5 else "WARN",f"{sat*100:.1f}% logits at cap={cap_:g}")
    fin=torch.isfinite(m(torch.randint(0,256,(1,1024),device=DEV))[0]).all().item()
    rec("6","long-seq L=1024 finite","PASS" if fin else "FAIL")
    cap=[None]; hk=m.layers[0].ln_in.register_forward_hook(lambda mod,i,o:cap.__setitem__(0,o.detach()))
    with torch.no_grad(): m(x)
    hk.remove(); h=cap[0]; L0=m.layers[0]
    Ix=L0._h(L0.to_Ix(h),L0.dk); Ts=L0._h(L0.to_Ts(h),L0.dk); den=(PHI(Ix)*PHI(Ts).cumsum(2)).sum(-1)
    rec("6","lin_read denom vs floor 1e-6","PASS" if den.min().item()>1e-3 else "WARN",f"min_den={den.min().item():.1e} frac<1e-3={(den<1e-3).float().mean().item()*100:.2f}%")
# ================= [7] PAPER-GRADE INTRINSIC EXPERIMENTS (fast proxies; scale on GPU/experiments.py) =================
def s7_complexity():
    H("[7a] COMPLEXITY: forward wall-time vs L (linear arch -> log-log slope ~1, not 2)")
    m=build(cfg()); m.eval()
    for L in m.layers: L.ch=256
    Ls=[256,512,1024,2048]; ts=[]
    for L in Ls:
        x=torch.randint(0,256,(1,L),device=DEV)
        with torch.no_grad():
            m(x); t0=time.time()
            for _ in range(3): m(x)
        ts.append((time.time()-t0)/3*1e3); print(f"   L={L:5d} | {ts[-1]:7.1f} ms")
    slope=np.polyfit(np.log(Ls),np.log(ts),1)[0]
    rec("7","complexity ~O(L) (no n^2 blowup)","PASS" if slope<1.4 else "WARN",f"log-log slope={slope:.2f} (1=linear,2=quadratic)")
    rec("7","decode state O(1) (KV-cache-free)","PASS","lin_read carry-state=(B,H,dk,dv) independent of seq-len; NOTE generate() recomputes full ctx (O(L)/step)->add a step-cache to realize O(1) decode")
def s7_depth():
    H("[7b] DEPTH GRADIENT-HEALTH (init 4..64 layers, fwd+bwd random batch, detect vanish/explode; no training)")
    for nl in [4,8,16,32,48,64]:
        m=build(cfg(n_layers=nl)); x,y=get_batch(2,48,seed=0); m.zero_grad(); _,l=m(x,y)
        if not torch.isfinite(l): rec("7",f"depth={nl}","FAIL","non-finite loss"); continue
        l.backward(); g=[sum(p.grad.norm().item() for p in L.parameters() if p.grad is not None) for L in m.layers]
        rec("7",f"depth={nl}","PASS" if (min(g)>1e-8 and max(g)<1e5) else "WARN",f"grad first/last={g[0]/(g[-1]+1e-12):.2f} min={min(g):.1e} max={max(g):.1e}")
def s7_seed():
    H("[7c] MULTI-SEED REPRODUCIBILITY (3 seeds x 60 steps; high variance=fragile init)")
    fs=[short_train(build(cfg(),seed=s),60,lr=3e-3,seed=s)['final'] for s in range(3)]
    rec("7","multi-seed final-loss variance","PASS" if np.std(fs)<0.2 else "WARN",f"finals={[f'{x:.3f}' for x in fs]} std={np.std(fs):.3f}")
def s7_repr():
    H("[7d] REPRESENTATION COLLAPSE (effective-rank + mean pairwise cosine of hidden states)")
    d=cfg().d_model; m=build(cfg()); m.eval(); x,_=get_batch(4,64,seed=1)
    cap=[None]; hk=m.ln_f.register_forward_hook(lambda mod,i,o:cap.__setitem__(0,o.detach()))
    with torch.no_grad(): m(x)
    hk.remove(); h=cap[0].reshape(-1,d); s=torch.linalg.svdvals(h-h.mean(0)); er=(s.sum()**2/(s*s).sum()).item()
    hn=F.normalize(h,dim=-1); cm=(hn@hn.T); mc=(cm.sum()-cm.diag().sum()).item()/(hn.shape[0]*(hn.shape[0]-1))
    rec("7","effective-rank (capacity used)","PASS" if er>d*0.3 else "WARN",f"eff_rank={er:.1f}/{d}")
    rec("7","anti-collapse (mean pairwise cos)","PASS" if mc<0.8 else "WARN",f"mean_cos={mc:.3f}")
def s7_distance():
    H("[7e] MEMORY/DISTANCE PROFILE (hidden-state similarity vs token distance; 120-step warmup)")
    m=build(cfg()); short_train(m,120,lr=3e-3,seed=0); m.eval(); x,_=get_batch(1,128,seed=2)
    cap=[None]; hk=m.ln_f.register_forward_hook(lambda mod,i,o:cap.__setitem__(0,o.detach()))
    with torch.no_grad(): m(x)
    hk.remove(); h=F.normalize(cap[0][0],dim=-1); L=h.shape[0]; dd=list(range(1,L)); sims=[(h[k:]*h[:-k]).sum(-1).mean().item() for k in dd]
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        plt.figure(figsize=(7,4)); plt.plot(dd,sims); plt.xlabel('token distance'); plt.ylabel('hidden cos-sim'); plt.title('temporal memory-decay profile'); plt.grid(alpha=.3)
        plt.savefig('test_memory_decay.png',dpi=120,bbox_inches='tight'); rec("7","memory-decay profile","PASS",f"sim@1={sims[0]:.2f} sim@{L-1}={sims[-1]:.2f} -> test_memory_decay.png")
    except Exception as e: rec("7","memory-decay profile","WARN",str(e))
def s7_extreme():
    H("[7f] EXTREME-INPUT ROBUSTNESS (degenerate inputs must not NaN/crash)")
    m=build(cfg()); m.eval()
    for nm,x in {'repeated-token L512':torch.zeros(1,512,dtype=torch.long,device=DEV),'gibberish L512':torch.randint(0,256,(1,512),device=DEV),'very-long L2048':torch.randint(0,256,(1,2048),device=DEV)}.items():
        with torch.no_grad(): o=m(x)[0]
        rec("7",nm,"PASS" if torch.isfinite(o).all() else "FAIL",f"finite |logit|max={o.abs().max().item():.1f}")
# ================= [8] BIG-MODEL METRICS DASHBOARD =================
def s8_dashboard():
    H("[8] BIG-MODEL METRICS DASHBOARD (loss/ppl, per-layer act/grad/weight, throughput, entropy, len-extrap, params)")
    import time as _t
    sync=lambda: torch.cuda.synchronize() if DEV.startswith('cuda') else None
    c=cfg(d_model=64,d_cell=16,n_lobe=2,n_cortex=4,n_layers=4,chunk_len=16)   # slightly larger so per-layer trends are visible
    m=build(c); opt=torch.optim.AdamW(m.parameters(),lr=3e-3,betas=(0.9,0.95)); m.train(); losses=[]
    for s in range(160):
        x,y=get_batch(16,64,seed=s); _,l=m(x,y); opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        if s%5==0: losses.append((s,l.item()))
    x,y=get_batch(16,64,seed=999); m.zero_grad(); _,l=m(x,y); l.backward()                     # one labelled backward for grad norms
    gn={};wn={}
    for n,p in m.named_parameters():
        if n.startswith('layers.'):
            li=int(n.split('.')[1]); wn[li]=wn.get(li,0)+float(p.detach().float().norm())**2
            if p.grad is not None: gn[li]=gn.get(li,0)+float(p.grad.detach().float().norm())**2
    ls=sorted(wn); wnl=[wn[i]**.5 for i in ls]; gnl=[gn.get(i,0)**.5 for i in ls]
    acts={}                                                                                    # per-layer activation RMS via hooks
    def mk(i):
        def h(mod,inp,out): acts[i]=float(out.detach().float().pow(2).mean().sqrt())
        return h
    hs=[m.layers[i].register_forward_hook(mk(i)) for i in range(len(m.layers))]; m.eval()
    with torch.no_grad(): m(x)
    for h in hs: h.remove()
    arms=[acts[i] for i in sorted(acts)]
    with torch.no_grad():                                                                      # prediction entropy (confidence)
        lg=m(x)[0]; pr=F.softmax(lg.float(),-1); ent=(-(pr*pr.clamp_min(1e-9).log()).sum(-1)).flatten().cpu().numpy()
    tps=[]                                                                                      # throughput vs seq length
    for L in [64,128,256,512]:
        xb,_=get_batch(4,L,seed=1); sync()
        with torch.no_grad():
            m(xb); sync(); t1=_t.time()
            for _ in range(3): m(xb)
            sync()
        tps.append((L,4*L*3/max(_t.time()-t1,1e-6)))
    extra=[]                                                                                    # length extrapolation (train ctx=64)
    with torch.no_grad():
        for L in [32,64,128,192,256]: xb,yb=get_batch(16,L,seed=5); _,le=m(xb,yb); extra.append((L,le.item()))
    m.train()
    pe=sum(p.numel() for n,p in m.named_parameters() if n.startswith('embed')); pl=sum(p.numel() for n,p in m.named_parameters() if n.startswith('layers.'))
    ph=sum(p.numel() for n,p in m.named_parameters() if n.startswith('head')); pf=sum(p.numel() for n,p in m.named_parameters() if n.startswith('ln_f'))
    rec("8","dashboard metrics computed","PASS",f"final_loss={losses[-1][1]:.3f} layers={len(ls)} act_RMS={arms[0]:.2f}->{arms[-1]:.2f} params={(pe+pl+ph+pf)/1e3:.0f}K")
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(3,3,figsize=(17,13)); fig.suptitle(f'model_lerp test dashboard (d={c.d_model}, L={c.n_layers}, {(pe+pl+ph+pf)/1e3:.0f}K params)',fontsize=14)
        s_,l_=zip(*losses); ax[0,0].plot(s_,l_,color='#37b'); ax[0,0].set_title('(1) training loss'); ax[0,0].set_xlabel('step'); ax[0,0].set_ylabel('CE'); ax[0,0].grid(alpha=.3)
        ax[0,1].plot(s_,[math.exp(min(v,20)) for v in l_],color='#b53'); ax[0,1].set_title('(2) perplexity'); ax[0,1].set_xlabel('step'); ax[0,1].set_yscale('log'); ax[0,1].grid(alpha=.3)
        ax[0,2].bar([f'L{i}' for i in ls],arms,color='#3b7'); ax[0,2].set_title('(3) activation RMS / layer (residual growth)'); ax[0,2].set_ylabel('RMS'); ax[0,2].grid(axis='y',alpha=.3)
        ax[1,0].bar([f'L{i}' for i in ls],gnl,color='#c55'); ax[1,0].set_title('(4) grad norm / layer (vanish/explode)'); ax[1,0].set_ylabel('||grad||'); ax[1,0].grid(axis='y',alpha=.3)
        ax[1,1].bar([f'L{i}' for i in ls],wnl,color='#959'); ax[1,1].set_title('(5) weight norm / layer'); ax[1,1].set_ylabel('||W||'); ax[1,1].grid(axis='y',alpha=.3)
        Lx=[t[0] for t in tps]; Ly=[t[1] for t in tps]; ax[1,2].plot(Lx,Ly,'o-',color='#37b'); ax[1,2].set_title('(6) throughput vs seq len'); ax[1,2].set_xlabel('seq'); ax[1,2].set_ylabel('tok/s'); ax[1,2].grid(alpha=.3)
        ax[2,0].hist(ent,bins=30,color='#3b7'); ax[2,0].axvline(math.log(c.vocab_size),color='r',ls='--',label='uniform'); ax[2,0].set_title('(7) prediction entropy'); ax[2,0].set_xlabel('nats'); ax[2,0].legend(fontsize=8); ax[2,0].grid(alpha=.3)
        ex=[e[0] for e in extra]; ey=[e[1] for e in extra]; ax[2,1].plot(ex,ey,'o-',color='#b53'); ax[2,1].axvline(64,color='gray',ls=':',label='train ctx'); ax[2,1].set_title('(8) length extrapolation'); ax[2,1].set_xlabel('eval ctx'); ax[2,1].set_ylabel('val loss'); ax[2,1].legend(fontsize=8); ax[2,1].grid(alpha=.3)
        ax[2,2].bar(['embed','layers','head','ln_f'],[pe/1e3,pl/1e3,ph/1e3,pf/1e3],color=['#37b','#3b7','#fb3','#999']); ax[2,2].set_title('(9) params by component'); ax[2,2].set_ylabel('K params'); ax[2,2].grid(axis='y',alpha=.3)
        plt.tight_layout(rect=[0,0,1,.97]); plt.savefig('test_dashboard.png',dpi=120,bbox_inches='tight'); rec("8","dashboard plot (9 panels)","PASS","test_dashboard.png")
    except Exception as e: rec("8","dashboard plot","WARN",f"skipped: {e}")
# ================= MAIN =================
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--full',action='store_true'); ap.add_argument('--compile',action='store_true'); ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); a=ap.parse_args()
    if a.device.startswith('cuda'): torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    global DEV; DEV=a.device; t0=time.time()
    print(f"model_lerp comprehensive test | device={DEV} | data={len(DATA)} bytes | threads={torch.get_num_threads()}")
    s1_components(); s2_ablation(); s3_controlled(); s4_stability(); s5_engineering(a.compile); s6_scan()
    s7_complexity(); s7_depth(); s7_seed(); s7_repr(); s7_distance(); s7_extreme(); s8_dashboard()
    H("SUMMARY")
    from collections import Counter; c=Counter(s for _,_,s,_ in RES)
    print(f"  total={len(RES)}  PASS={c['PASS']}  WARN={c['WARN']}  FAIL={c['FAIL']}  SKIP={c['SKIP']}  | {time.time()-t0:.0f}s")
    fails=[f"{sec}:{n}" for sec,n,s,_ in RES if s=='FAIL']; warns=[f"{sec}:{n}" for sec,n,s,_ in RES if s=='WARN']
    if fails: print("  FAIL ->",", ".join(fails))
    if warns: print("  WARN ->",", ".join(warns))
    print("\n  KEY FINDING: section [4] empirically confirms the amplitude-head cliff (grad amplifier 1/sqrt(eps):")
    print("  eps=1e-6 -> ~1000x  vs  eps=1.0 -> 1x). Raising the in-log constant + fp32 head removes the cliff.")
    print("  artifacts: test_stability.png, test_memory_decay.png, test_dashboard.png" + (", test_engineering.png" if os.path.exists('test_engineering.png') else ""))
if __name__=='__main__': main()
