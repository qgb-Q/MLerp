"""ablation.py -- recall ablations for model_lerp. The open question NOW is DELTA-RULE (DeltaNet erase-before-write
vs additive); phi-expand is ALREADY CONCLUDED (A5000 5-seed: m=64 phi2=0.598 vs phi1=0.262, +0.336, t=3.84 p<0.01
-> beneficial) and is NOT re-run by default (--exp phi to re-verify).
Other CONCLUDED experiments are also not re-run: hop2(R_a)/T_a redundant on tested tasks (drops within noise);
conv>=lin self-think; pathway-contribution measured (layer0 ||Xl2||>||Xl1|| but ablation-neutral = used-but-substitutable).
SPEED NOTE: each "model" is tiny -> KERNEL-LAUNCH-BOUND on the GPU (hundreds of small kernels/step). --compile (TorchInductor)
*can* fuse them, BUT on Windows it often crashes ptxas (RuntimeError: ptxas failed with error code 3221225786, empty stderr)
on inductor-generated kernels -- a known Windows toolchain bug, NOT the model (hand-written Triton kernels compile fine). So
do NOT rely on --compile here. The RELIABLE speedup is fewer model-trainings: cut --seeds and --m_list. delta now uses the
CHUNKED-PARALLEL DeltaNet (WY form, O(L/chunk) sequential -- verified == sequential ref + gradcheck), so use_delta is only
mildly slower than additive (the per-token O(L) loop that made it crawl is gone).
Run (A5000): python ablation.py --exp delta                          # delta only; conclusive (5 seeds, m=[16,48,64]) = 30 models
  quick first look: python ablation.py --exp delta --seeds 3 --m_list 64    # 6 models, the dense-key regime that matters
"""
import argparse,statistics as st
import numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from model_lerp import LerpConfig,LerpLM
NV=70; IGN=-100; QRY=NV; VOCAB=NV+1; CHANCE=1.0/NV                      # NV=70 -> m up to 64 (> 2x key-dim) to stress phi1; answers are value tokens in [0,NV)
def t_recall(B,m,dev):                                                  # k1 v1 .. km vm QRY kq -> vq. dense keys stress key separability. GPU-NATIVE + vectorized.
    keys=torch.rand(B,NV,device=dev).argsort(-1)[:,:m]                  # B distinct-key rows in ONE kernel (replaces B python randperm launches)
    vals=torch.randint(0,NV,(B,m),device=dev); ar=torch.arange(B,device=dev); qi=torch.randint(0,m,(B,),device=dev)
    kq=keys[ar,qi]; vq=vals[ar,qi]; pairs=torch.stack([keys,vals],-1).reshape(B,2*m)
    x=torch.cat([pairs,torch.full((B,1),QRY,device=dev),kq[:,None],vq[:,None]],1)
    L=x.shape[1]; y=torch.full((B,L),IGN,device=dev); y[:,L-2]=vq; return x,y
@torch.no_grad()
def acc(net,x,y):
    net.eval(); pred=net(x)[0].argmax(-1); mask=(y!=IGN); r=((pred==y)|~mask).all(1).float().mean().item(); net.train(); return r
def fit(m,var,seed):
    torch.manual_seed(seed)
    kw=dict(vocab_size=VOCAB,d_model=A.d_model,d_cell=A.d_cell,n_lobe=2,n_cortex=A.n_cortex,n_layers=A.n_layers,chunk_len=32,phi_expand=2); kw.update(var)
    net=LerpLM(LerpConfig(**kw)).to(A.device); net.train(); opt=torch.optim.AdamW(net.parameters(),lr=A.lr,betas=(0.9,0.95),weight_decay=0.01)
    run=torch.compile(net) if A.compile else net; amp=A.device.startswith('cuda'); torch.manual_seed(seed*7+1)
    xe,ye=t_recall(256 if amp else 96,m,A.device); best=0.0; bad=0; stop=A.steps                   # fixed eval batch -> stable early-stop signal
    for step in range(1,A.steps+1):
        x,y=t_recall(A.bs,m,A.device)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp): _,loss=run(x,y)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
        if step%A.eval_every==0:                                                                   # EARLY-STOP: recall plateaus well before --steps; stop at plateau/near-perfect -> no fixed-step waste. eval on UNCOMPILED net (shares params) so the 256-batch doesn't trigger a recompile
            r=acc(net,xe,ye)
            if r>best+5e-3: best=r; bad=0
            else: bad+=1
            if best>=0.99 or bad>=A.patience: stop=step; break
    return best,stop
def seeds_mean(m,var):
    rs=[fit(m,var,s) for s in range(A.seeds)]; a=[r[0] for r in rs]
    return sum(a)/len(a),(st.pstdev(a) if len(a)>1 else 0.0),sum(r[1] for r in rs)/len(rs)
def sweep(title,vA,vB,labA,labB):                                       # generic A-vs-B over m_list; significance = SE-of-difference (not 2*max sigma; a variant's collapse-variance is signal, not a threshold)
    print(f"\n=== {title} ===")
    rows=[]
    for m in A.m_list:
        pA,sA,stA=seeds_mean(m,vA); pB,sB,stB=seeds_mean(m,vB); d=pA-pB; se=(sA*sA/A.seeds+sB*sB/A.seeds)**0.5; sig=2*max(se,.005); t=d/max(se,1e-9)
        learned=max(pA,pB)>max(CHANCE*3,0.10)
        tag=('@chance(model too small)' if not learned else (f'{labA} SIGNIFICANT (t={t:.1f})' if d>sig else (f'{labB} better (t={t:.1f})' if -d>sig else f'tie (t={t:.1f})')))
        rows.append((m,pA,sA,pB,sB,labA,labB)); print(f"  m={m:3d}: {labA}={pA:.3f}+/-{sA:.3f}  {labB}={pB:.3f}+/-{sB:.3f}  delta={d:+.3f}  {tag}  [~{int((stA+stB)/2)} steps]")
    return rows
SCANKEY={'phi':'phi_expand','cortex':'n_cortex','dcell':'d_cell'}                        # key-separability / capacity knobs (ADDITIVE only -- delta is dead on distinct-key recall). recall ~ logistic(log2(H*dc^2)); phi adds the separability axis dk=phi*dc
def scan(knob,vals):                                                                     # additive recall vs one knob across m_list -> how far past the ~0.6 dense-key collision cap can we push, and where does it saturate?
    key=SCANKEY[knob]; print(f"\n=== SEPARABILITY/CAPACITY scan: {knob}({key}), ADDITIVE, recall vs m ===")
    table={}
    for m in A.m_list:
        row=[(v,)+seeds_mean(m,{key:v}) for v in vals]                                    # {key:v} override; use_delta stays False -> (v,mean,std,stop)
        table[m]=row; best=max(row,key=lambda r:r[1]); avgstep=int(sum(r[3] for r in row)/len(row))
        print(f"  m={m:3d}: "+"  ".join(f"{key}={v}:{p:.3f}+/-{s:.3f}" for v,p,s,_ in row)+f"   -> best {key}={best[0]} ({best[1]:.3f})  [~{avgstep} steps]")
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; plt.figure(figsize=(7,4.5))
        for m,row in table.items(): plt.errorbar([r[0] for r in row],[r[1] for r in row],yerr=[r[2] for r in row],marker='o',capsize=4,label=f'm={m}')
        plt.axhline(CHANCE,ls=':',c='r',alpha=.5,label='chance'); plt.xlabel(key); plt.ylabel('recall acc'); plt.ylim(0,1); plt.title(f'recall vs {key} (additive, seeds={A.seeds}, dc={A.d_cell})'); plt.legend(); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f'ablation_scan_{knob}.png',dpi=120); print(f"[done] -> ablation_scan_{knob}.png")
    except Exception as e: print(f"[plot skipped] {e}")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seeds',type=int,default=5); p.add_argument('--steps',type=int,default=2000); p.add_argument('--lr',type=float,default=3e-3); p.add_argument('--bs',type=int,default=64)
    p.add_argument('--eval_every',type=int,default=200); p.add_argument('--patience',type=int,default=4)   # EARLY-STOP: eval recall every N steps; stop after `patience` evals w/o improvement (or recall>=0.99) -> trains only to convergence, not a fixed 2000
    p.add_argument('--d_model',type=int,default=128); p.add_argument('--d_cell',type=int,default=32); p.add_argument('--n_cortex',type=int,default=8); p.add_argument('--n_layers',type=int,default=3)
    p.add_argument('--m_list',type=int,nargs='+',default=[16,48,64]); p.add_argument('--compile',action='store_true'); p.add_argument('--smoke',action='store_true')
    p.add_argument('--scan',choices=['phi','cortex','dcell'],default='phi'); p.add_argument('--scan_vals',type=int,nargs='+',default=None)   # for --exp sep
    p.add_argument('--exp',choices=['phi','delta','both','sep'],default='delta')   # delta=CONCLUDED (REFUTED on distinct-key recall, see 10.11). sep=key-separability/capacity scan (additive) -- the open recall lever. phi=re-verify phi-expand.
    A=p.parse_args()
    if A.device.startswith('cuda'): torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    if A.smoke: A.seeds=1; A.steps=40; A.d_model=48; A.d_cell=16; A.n_cortex=4; A.n_layers=2; A.m_list=[16,32]; A.eval_every=20; A.patience=2
    H=2*A.n_cortex; kd1=A.d_cell; kd2=2*A.d_cell
    print(f"ABLATION | exp={A.exp} device={A.device} seeds={A.seeds} steps={A.steps} model(d={A.d_model},dc={A.d_cell},H={H},L={A.n_layers}) chance={CHANCE:.3f}")
    print(f"  phi1 key-dim={kd1} phi2 key-dim={kd2} (phi1 collides once m>{kd1}); delta=erase-before-write (chunked-parallel) vs additive")
    res={}
    if A.exp=='sep':
        if A.scan_vals is None: A.scan_vals={'phi':[2,3,4],'cortex':[4,8,16,32],'dcell':[16,32,48,64]}[A.scan]   # phi=1 already known worst (collides), skip
        scan(A.scan,A.scan_vals)                                                          # makes its own ablation_scan_<knob>.png
    if A.exp in ('phi','both'):  res['phi']  =sweep(f"PHI-EXPAND conclusiveness (phi2 key-dim {kd2} vs phi1 {kd1})",{'phi_expand':2},{'phi_expand':1},'phi2','phi1')
    if A.exp in ('delta','both'):res['delta']=sweep("DELTA-RULE recall: DeltaNet erase-before-write vs additive (the memory upgrade)",{'use_delta':True},{'use_delta':False},'delta','additive')
    if res:
      try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,len(res),figsize=(6*len(res),4.5),squeeze=False); axes=axes[0]
        for ax,(name,rows) in zip(axes,res.items()):
            ms=[r[0] for r in rows]; labA,labB=rows[0][5],rows[0][6]
            ax.errorbar(ms,[r[1] for r in rows],yerr=[r[2] for r in rows],marker='o',capsize=4,label=labA)
            ax.errorbar(ms,[r[3] for r in rows],yerr=[r[4] for r in rows],marker='s',capsize=4,label=labB)
            ax.axhline(CHANCE,ls=':',c='r',alpha=.5,label='chance'); ax.axvline(kd1,ls='--',c='gray',alpha=.5)
            ax.set_xlabel('m (key-value pairs)'); ax.set_ylabel('recall acc'); ax.set_title(f'{name} (seeds={A.seeds},dc={A.d_cell})'); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig('ablation_summary.png',dpi=120); print("[done] -> ablation_summary.png")
      except Exception as e: print(f"[plot skipped] {e}")
