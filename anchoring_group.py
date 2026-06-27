"""anchoring_group.py -- CONTROL/ANCHOR experiment. A STANDARD Transformer (GPT-2 style: pre-LN, SDPA attention,
KV-cache decode), matched to model_lerp on EVERY confound, to test whether the brain-inspired architecture actually wins.
MATCHED to train.py: param count, tokenizer, training data (.bin), token budget, batch/accum/context, AdamW(0.9,0.95)
+ WD policy (no-decay on <2D params), warmup+cosine LR, grad-clip, AMP/dtype, random seed, #steps. The ONLY free knob
is FFN width (d_ff), auto-solved so params match at FIXED d_model / n_layers / vocab / context (FFN is the standard
capacity knob). Fair inference: GPT uses a KV-cache (its efficient form) vs model_lerp's O(1)/token incremental decode.
  train:    python anchoring_group.py                              # -> out_anchor/ckpt.pt ; pass same flags you'd pass train.py
  compare:  python anchoring_group.py --compare out_lerp/ckpt.pt   # side-by-side: params/val-loss/ppl/train tok-s/infer tok-s/VRAM/sample
"""
import os,sys,math,time,argparse,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from train import get_tok,make_batch
class Attn(nn.Module):
    def __init__(s,d,nh): super().__init__(); s.nh=nh; s.hd=d//nh; s.qkv=nn.Linear(d,3*d); s.proj=nn.Linear(d,d)
    def forward(s,x,cache=None):
        B,T,C=x.shape; q,k,v=s.qkv(x).split(C,2)
        q=q.view(B,T,s.nh,s.hd).transpose(1,2); k=k.view(B,T,s.nh,s.hd).transpose(1,2); v=v.view(B,T,s.nh,s.hd).transpose(1,2)
        if cache is not None: pk,pv=cache; k=torch.cat([pk,k],2); v=torch.cat([pv,v],2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=(cache is None and T>1))           # prefill: causal; single-token step: attend full cache
        return s.proj(y.transpose(1,2).reshape(B,T,C)),(k,v)
class Block(nn.Module):
    def __init__(s,d,nh,dff): super().__init__(); s.ln1=nn.LayerNorm(d); s.attn=Attn(d,nh); s.ln2=nn.LayerNorm(d); s.fc=nn.Linear(d,dff); s.pj=nn.Linear(dff,d)
    def forward(s,x,cache=None): a,nc=s.attn(s.ln1(x),cache); x=x+a; x=x+s.pj(F.gelu(s.fc(s.ln2(x)))); return x,nc
class GPT(nn.Module):                                                                       # standard pre-LN GPT, the anchor baseline
    def __init__(s,vocab,d,nh,L,dff,ctx,tie=True):
        super().__init__(); s.ctx=ctx; s.tok=nn.Embedding(vocab,d); s.pos=nn.Embedding(ctx,d)
        s.blocks=nn.ModuleList([Block(d,nh,dff) for _ in range(L)]); s.lnf=nn.LayerNorm(d); s.head=nn.Linear(d,vocab,bias=False)
        if tie: s.head.weight=s.tok.weight
        s.apply(s._init)
    def _init(s,m):
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight,0,0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0,0.02)
    def forward(s,idx,targets=None,grad_checkpoint=False):
        B,T=idx.shape; x=s.tok(idx)+s.pos(torch.arange(T,device=idx.device))[None]
        for b in s.blocks: x=(torch.utils.checkpoint.checkpoint(lambda y,bl=b:bl(y)[0],x,use_reentrant=False) if grad_checkpoint else b(x)[0])
        logits=s.head(s.lnf(x)); loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1)) if targets is not None else None
        return logits,loss
    def num_params(s): return sum(p.numel() for p in s.parameters())                        # .parameters() de-dups the tied head weight
    @torch.no_grad()
    def generate(s,idx,max_new,temperature=1.0,top_k=None):                                  # KV-cache decode (fair vs model_lerp incremental): prefill once, O(L)/token
        caches=[None]*len(s.blocks); T0=idx.size(1); x=s.tok(idx)+s.pos(torch.arange(T0,device=idx.device))[None]
        for i,b in enumerate(s.blocks): x,caches[i]=b(x,caches[i])
        logits=s.head(s.lnf(x))[:,-1]; out=idx
        for t in range(max_new):
            l=logits/max(temperature,1e-6)
            if top_k: v,_=torch.topk(l,min(top_k,l.size(-1))); l=l.masked_fill(l<v[:,[-1]],float('-inf'))
            nxt=torch.multinomial(F.softmax(l,-1),1); out=torch.cat([out,nxt],1); p=T0+t
            if p>=s.ctx: break                                                               # Transformer is context-bounded (model_lerp is not -- an architectural difference)
            x=s.tok(nxt)+s.pos(torch.tensor([p],device=idx.device))[None]
            for i,b in enumerate(s.blocks): x,caches[i]=b(x,caches[i])
            logits=s.head(s.lnf(x))[:,-1]
        return out
def match_dff(P,V,d,L,ctx):                                                                 # solve FFN width so total params == P at fixed d/L/vocab/ctx (tied embeddings). EXACT (per-param accounting); integer-rounding residual ~0.004%.
    fixed=V*d+ctx*d+2*d+L*(4*d*d+9*d); per=L*(2*d+1); return max(1,round((P-fixed)/per))
def load_data(bin_dir,tok):
    tr_bin=os.path.join(bin_dir,f'train_{tok.name}.bin'); va_bin=os.path.join(bin_dir,f'val_{tok.name}.bin')
    if not os.path.exists(tr_bin): print(f"[fatal] {tr_bin} not found -- the anchor must use the SAME .bin as train.py. Run prepare_data.py / train.py first, or set --bin_dir."); sys.exit(1)
    dt=np.uint8 if tok.vocab_size<=256 else np.uint16; tr=np.memmap(tr_bin,dtype=dt,mode='r'); n_tr=len(tr)
    if os.path.exists(va_bin) and os.path.getsize(va_bin)>0: va=np.memmap(va_bin,dtype=dt,mode='r')
    else: cut=int(n_tr*0.99); va=tr[cut:]; tr=tr[:cut]; n_tr=len(tr)
    return tr,n_tr,va,len(va)
def make_opt(model,wd,lr,cuda):                                                             # SAME policy as train.py: decay >=2D params, no-decay <2D (biases, LayerNorm)
    decay=[p for _,p in model.named_parameters() if p.dim()>=2]; nod=[p for _,p in model.named_parameters() if p.dim()<2]
    grp=[{'params':decay,'weight_decay':wd},{'params':nod,'weight_decay':0.0}]
    try: return torch.optim.AdamW(grp,lr=lr,betas=(0.9,0.95),fused=cuda)
    except TypeError: return torch.optim.AdamW(grp,lr=lr,betas=(0.9,0.95))
def train_anchor(a):
    os.makedirs(a.out_dir,exist_ok=True); torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev=a.device; cuda=dev.startswith('cuda')
    if cuda: torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    tok=get_tok(a.tokenizer); tr,n_tr,va,n_va=load_data(a.bin_dir,tok); splits={'train':(tr,n_tr),'val':(va,n_va)}
    print(f"[cfg] tokenizer={tok.name} vocab={tok.vocab_size} device={dev} dtype={a.dtype}")
    print(f"[data] (SHARED with train.py) train={n_tr/1e6:.2f}M val={n_va/1e6:.2f}M tokens from {a.bin_dir}")
    dff=match_dff(a.match_params,tok.vocab_size,a.d_model,a.n_layers,a.block_size)
    model=GPT(tok.vocab_size,a.d_model,a.n_heads,a.n_layers,dff,a.block_size,tie=not a.untie).to(dev)
    print(f"[anchor] STANDARD Transformer | params={model.num_params()/1e6:.2f}M (target {a.match_params/1e6:.2f}M, d_ff auto={dff}) | d={a.d_model} heads={a.n_heads} layers={a.n_layers} ctx={a.block_size} tie={not a.untie}")
    if a.compile: model=torch.compile(model); print("[anchor] torch.compile enabled")
    opt=make_opt(model,a.weight_decay,a.lr,cuda); amp=(a.dtype!='fp32') and cuda; adt=torch.bfloat16 if a.dtype=='bf16' else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=(a.dtype=='fp16' and cuda)); ckpt_path=os.path.join(a.out_dir,'ckpt.pt'); step=0; best=float('inf')
    def lr_at(it):                                                                          # IDENTICAL schedule to train.py
        if it<a.warmup: return a.lr*(it+1)/a.warmup
        if it>=a.max_steps: return a.min_lr
        r=(it-a.warmup)/max(1,a.max_steps-a.warmup); return a.min_lr+0.5*(a.lr-a.min_lr)*(1+math.cos(math.pi*r))
    @torch.no_grad()
    def evaluate():
        model.eval(); out={}
        for sp,(mm,n) in splits.items():
            ls=[make_batch(mm,n,a.batch_size,a.block_size,dev,cuda) for _ in range(a.eval_iters)]
            vv=[]
            for x,y in ls:
                with torch.autocast('cuda',dtype=adt,enabled=amp): _,l=model(x,y)
                vv.append(l.item())
            out[sp]=sum(vv)/len(vv)
        model.train(); return out
    print(f"[train] start max={a.max_steps} bs={a.batch_size} seq={a.block_size} accum={a.grad_accum} eff_tok/step={a.batch_size*a.block_size*a.grad_accum} seed={a.seed}")
    model.train(); t0=time.time(); tlog=t0
    if cuda: torch.cuda.reset_peak_memory_stats()
    while step<a.max_steps:
        lr=lr_at(step)
        for g in opt.param_groups: g['lr']=lr
        opt.zero_grad(set_to_none=True); lacc=0.0
        for _ in range(a.grad_accum):
            x,y=make_batch(splits['train'][0],splits['train'][1],a.batch_size,a.block_size,dev,cuda)
            with torch.autocast('cuda',dtype=adt,enabled=amp): _,loss=model(x,y,grad_checkpoint=a.grad_ckpt); loss=loss/a.grad_accum
            scaler.scale(loss).backward(); lacc+=loss.item()
        if a.grad_clip>0: scaler.unscale_(opt); gn=float(nn.utils.clip_grad_norm_(model.parameters(),a.grad_clip))
        else: gn=0.0
        scaler.step(opt); scaler.update(); step+=1
        if step%a.log_interval==0:
            d2=time.time()-tlog; tps=a.batch_size*a.block_size*a.grad_accum*a.log_interval/max(d2,1e-9); tlog=time.time(); vram=torch.cuda.max_memory_allocated()/1e9 if cuda else 0
            print(f"step {step:6d} | loss {lacc:.4f} | gnorm {gn:6.2f} | lr {lr:.2e} | {tps:8.0f} tok/s | vram {vram:.1f}GB | {time.time()-t0:6.0f}s")
        if step%a.eval_interval==0 or step==a.max_steps:
            ev=evaluate(); print(f"  [eval] step {step} train {ev['train']:.4f} val {ev['val']:.4f}"+("  *best*" if ev['val']<best else ""))
            sd=(model._orig_mod if hasattr(model,'_orig_mod') else model).state_dict()
            ck={'model':sd,'opt':opt.state_dict(),'step':step,'tokenizer':tok.name,'best_val':min(best,ev['val']),'cfg':{'model_type':'gpt','vocab':tok.vocab_size,'d':a.d_model,'nh':a.n_heads,'L':a.n_layers,'dff':dff,'ctx':a.block_size,'tie':not a.untie}}
            torch.save(ck,ckpt_path)
            if ev['val']<best: best=ev['val']; torch.save(ck,os.path.join(a.out_dir,'ckpt_best.pt'))
    print(f"[done] anchor step={step} best_val={best:.4f} ckpt={ckpt_path} | peak VRAM {(torch.cuda.max_memory_allocated()/1e9 if cuda else 0):.2f}GB")
def load_lerp(path,dev):
    from model_lerp import LerpConfig,LerpLM
    ck=torch.load(path,map_location=dev,weights_only=False); sd={k.replace('_orig_mod.',''):v for k,v in ck['model'].items()}
    cfgd=dict(ck['cfg']); cfgd['head_linear_base']=('head.base.weight' in sd); cfgd['head_born']=any('head.to_re' in k for k in sd)
    m=LerpLM(LerpConfig(**cfgd)).to(dev); m.load_state_dict(sd); m.eval(); return m
def load_anchor(path,dev):
    ck=torch.load(path,map_location=dev,weights_only=False); c=ck['cfg']
    m=GPT(c['vocab'],c['d'],c['nh'],c['L'],c['dff'],c['ctx'],tie=c.get('tie',True)).to(dev); m.load_state_dict(ck['model']); m.eval(); return m
@torch.no_grad()
def val_loss(m,mm,N,dev,block,nb,bs,amp):
    step=block*bs; tot=0.0; cnt=0
    for off in np.linspace(0,max(N-step-1,1),nb).astype(np.int64):
        win=np.stack([mm[off+i*block:off+i*block+block+1] for i in range(bs)])
        x=torch.from_numpy(win[:,:-1].astype(np.int64)).to(dev); y=torch.from_numpy(win[:,1:].astype(np.int64)).to(dev)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp): _,l=m(x,y)
        tot+=l.item(); cnt+=1
    return tot/max(cnt,1)
def bench_train(m,dev,bs,block,vocab,amp,iters=5):
    x=torch.randint(0,vocab,(bs,block),device=dev); y=torch.randint(0,vocab,(bs,block),device=dev); opt=torch.optim.AdamW(m.parameters(),lr=1e-5)
    for _ in range(2):
        opt.zero_grad()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp): _,l=m(x,y)
        l.backward(); opt.step()
    if dev.startswith('cuda'): torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t=time.time()
    for _ in range(iters):
        opt.zero_grad()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp): _,l=m(x,y)
        l.backward(); opt.step()
    if dev.startswith('cuda'): torch.cuda.synchronize()
    return bs*block*iters/(time.time()-t),(torch.cuda.max_memory_allocated()/1e9 if dev.startswith('cuda') else 0)
@torch.no_grad()
def bench_infer_scan(m,dev,plen,lengths,vocab,amp,fast='auto'):                              # inference scan -> (xs=actual generated, tps=tok/s, vram=peak GB, lat=total s) per length. Transformer self-caps at its ctx.
    def gen(idx,Ln):
        try: return m.generate(idx,Ln,fast=fast)                                             # LerpLM: graph/compile/eager. GPT: no 'fast' kwarg -> fall back.
        except TypeError: return m.generate(idx,Ln)
    gen(torch.randint(0,vocab,(1,plen),device=dev),8)                                         # warmup
    xs=[]; tps=[]; vram=[]; lat=[]
    for Ln in lengths:
        idx=torch.randint(0,vocab,(1,plen),device=dev)
        if dev.startswith('cuda'): torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t=time.time()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp): out=gen(idx,Ln)
        if dev.startswith('cuda'): torch.cuda.synchronize()
        dt=time.time()-t; A=out.size(1)-plen                                                 # ACTUAL generated (GPT beyond ctx generates fewer -> curves stop at ctx)
        xs.append(A); tps.append(A/max(dt,1e-9)); lat.append(dt); vram.append(torch.cuda.max_memory_allocated()/1e9 if dev.startswith('cuda') else 0)
    return xs,tps,vram,lat
def compare_plot(D,out):
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import textwrap
    except Exception as e: print(f"[warn] plot skipped: {e}"); return
    names=D['names']; C=['#2c7fb8','#d95f0e']; w=0.36
    fig=plt.figure(figsize=(20,10)); fig.suptitle(f"model_lerp  vs  anchor (standard Transformer) — controlled comparison  |  matched: {D['meta']}",fontsize=12)
    def bars(ax,mets,title,fmt):
        x=np.arange(len(mets))
        for j,nm in enumerate(names):
            vals=[m[1][j] for m in mets]; norm=[v/max(m[1][0],m[1][1],1e-9) for v,m in zip(vals,mets)]
            ax.bar(x+(j-0.5)*w,norm,w,color=C[j],label=nm)
            for xi,v,nv in zip(x,vals,norm): ax.text(xi+(j-0.5)*w,nv+0.02,fmt(v),ha='center',va='bottom',fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels([m[0] for m in mets],fontsize=8); ax.set_title(title); ax.set_ylim(0,1.28); ax.legend(fontsize=8); ax.grid(alpha=.3,axis='y')
    def scan(ax,k,ylab,title,loglog=False):                                                  # k: index into scan tuple (1=tps,2=vram,3=lat)
        for j,nm in enumerate(names): ax.plot(D['scan'][j][0],D['scan'][j][k],'o-',color=C[j],lw=2,label=nm)
        ax.axvline(D['ctx'],color='gray',ls='--',lw=.8); ax.text(D['ctx'],ax.get_ylim()[1]*0.5,f" Transformer ctx={D['ctx']}",fontsize=7,color='gray',rotation=90,va='center')
        ax.set_xscale('log'); 
        if loglog: ax.set_yscale('log')
        ax.set_xlabel('tokens generated'); ax.set_ylabel(ylab); ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=.3,which='both')
    bars(fig.add_subplot(2,4,1),[('params (M)',D['params']),('train tok/s',D['train_tps']),('infer tok/s',D['infer_tps']),('VRAM (GB)',D['vram'])],'(1) resources & speed  (bar=relative, label=actual)',lambda v:f"{v:.0f}" if v>=10 else f"{v:.2f}")
    bars(fig.add_subplot(2,4,2),[('val loss (nats)',D['val_loss']),('val ppl',D['ppl'])],'(2) quality — LOWER is better',lambda v:f"{v:.2f}")
    axr=fig.add_subplot(2,4,3,projection='polar')                                            # (3) radar: outer = better
    axes=[('val loss',D['val_loss'],1),('ppl',D['ppl'],1),('train t/s',D['train_tps'],0),('infer t/s',D['infer_tps'],0),('VRAM',D['vram'],1),('params',D['params'],1)]
    def good(vals,lb):
        lo,hi=min(vals),max(vals)
        return [1.0]*len(vals) if hi==lo else [0.15+0.85*((hi-v)/(hi-lo) if lb else (v-lo)/(hi-lo)) for v in vals]
    ang=np.linspace(0,2*np.pi,len(axes),endpoint=False).tolist(); ang+=ang[:1]
    for j,nm in enumerate(names):
        vv=[good(ax[1],ax[2])[j] for ax in axes]; vv+=vv[:1]; axr.plot(ang,vv,color=C[j],lw=2,label=nm); axr.fill(ang,vv,color=C[j],alpha=.15)
    axr.set_xticks(ang[:-1]); axr.set_xticklabels([ax[0] for ax in axes],fontsize=8); axr.set_yticklabels([]); axr.set_title('(3) overall (outer = better)',pad=20)
    scan(fig.add_subplot(2,4,4),1,'tok/s','(4) inference THROUGHPUT vs length\nmodel_lerp O(1)/tok flat vs Transformer O(L) decay (+ctx-bounded)')
    scan(fig.add_subplot(2,4,5),2,'peak VRAM (GB)','(5) inference VRAM vs length\nmodel_lerp fixed state (flat) vs Transformer KV-cache (linear growth)')
    scan(fig.add_subplot(2,4,6),3,'total latency (s)','(6) inference TOTAL LATENCY vs length (log-log)\nmodel_lerp ~O(N) linear vs Transformer ~O(N^2) quadratic',loglog=True)
    for j,nm in enumerate(names):                                                            # (7),(8) samples
        ax=fig.add_subplot(2,4,7+j); ax.axis('off'); ax.set_title(f'({7+j}) sample — {nm}',fontsize=10,color=C[j])
        ax.text(0,1,textwrap.fill(D['samples'][j][:520],width=52),fontsize=7,va='top',family='monospace')
    plt.tight_layout(rect=[0,0,1,.95]); plt.savefig(out,dpi=120,bbox_inches='tight'); print(f"[plot] full comparison -> {out}")
def train_compare(a):
    dev=a.device; cuda=dev.startswith('cuda'); amp=cuda; tok=get_tok(a.tokenizer)
    anchor_path=a.anchor or os.path.join(a.out_dir,'ckpt.pt')
    if not os.path.exists(a.compare): print(f"[fatal] --compare ckpt not found: {a.compare}"); sys.exit(1)
    if not os.path.exists(anchor_path): print(f"[fatal] anchor ckpt not found: {anchor_path} (train it first: python anchoring_group.py)"); sys.exit(1)
    lerp=load_lerp(a.compare,dev); anch=load_anchor(anchor_path,dev); _,_,va,n_va=load_data(a.bin_dir,tok); models=[('model_lerp',lerp),('anchor(GPT)',anch)]
    lens=[int(x) for x in a.infer_scan.split(',')]
    D={'names':[nm for nm,_ in models],'params':[],'val_loss':[],'ppl':[],'train_tps':[],'infer_tps':[],'vram':[],'scan':[],'samples':[],'ctx':a.block_size,'meta':f"{tok.name} V={tok.vocab_size}, ctx={a.block_size}, bs={a.batch_size}, bf16={amp}"}
    print(f"\n=== CONTROLLED COMPARISON ({D['meta']}) ===")
    for nm,m in models:
        p=m.num_params()/1e6; L=val_loss(m,va,n_va,dev,a.block_size,a.eval_iters,a.batch_size,amp)
        ttps,tvram=bench_train(m,dev,a.batch_size,a.block_size,tok.vocab_size,amp); xs,tps,ivram,lat=bench_infer_scan(m,dev,min(16,a.block_size//4),lens,tok.vocab_size,amp,fast=a.lerp_decode)
        ix=min(range(len(lens)),key=lambda i:abs(lens[i]-512)); itps=tps[ix]                   # representative infer tok/s (~512 tok)
        smp=tok.decode(m.generate(torch.tensor([tok.encode(a.prompt) or [tok.eot]],device=dev),a.infer_tokens,temperature=0.8,top_k=50)[0].tolist())
        for k,v in [('params',p),('val_loss',L),('ppl',math.exp(L)),('train_tps',ttps),('infer_tps',itps),('vram',tvram),('scan',(xs,tps,ivram,lat)),('samples',smp)]: D[k].append(v)
        print(f"  {nm:<14} params={p:.2f}M | val_loss={L:.4f} ppl={math.exp(L):.2f} | train={ttps:.0f} t/s infer~512={itps:.0f} t/s | trainVRAM={tvram:.2f}GB inferVRAM@max={ivram[-1]:.2f}GB")
    print("\n  sample:")
    for nm,s in zip(D['names'],D['samples']): print(f"  [{nm}] {s!r}\n")
    if not a.no_plot: compare_plot(D,a.plot_out or os.path.join(a.out_dir,'comparison.png'))
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--compare',default=None,help='path to model_lerp ckpt -> run side-by-side comparison instead of training')
    p.add_argument('--anchor',default=None,help='(compare) anchor ckpt path; default <out_dir>/ckpt.pt')
    p.add_argument('--bin_dir',default='out_lerp',help='where train/val .bin live (SAME as train.py -> identical data)')
    p.add_argument('--out_dir',default='out_anchor',help='where to write anchor ckpts (separate from model_lerp)')
    p.add_argument('--match_params',type=int,default=253_220_000,help="model_lerp param count to match (from train.py's [model] line). 253M = single-read + no-born default; pass your actual count if you reallocated.")
    p.add_argument('--d_model',type=int,default=1024); p.add_argument('--n_heads',type=int,default=16); p.add_argument('--n_layers',type=int,default=12); p.add_argument('--untie',action='store_true',help='untie input/output embeddings (-> smaller d_ff)')
    p.add_argument('--tokenizer',default='neo'); p.add_argument('--batch_size',type=int,default=2); p.add_argument('--block_size',type=int,default=2048); p.add_argument('--grad_accum',type=int,default=4)   # defaults match train.py (bs2/block2048/accum4 = 16384 eff_tok) for a fair compute comparison -- keep them identical. block_size = Transformer ctx (HARD cap via learned pos emb); match_dff auto-shrinks d_ff to stay param-matched. SDPA=flash -> attn mem O(L) not O(L^2).
    p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--min_lr',type=float,default=3e-5); p.add_argument('--warmup',type=int,default=500); p.add_argument('--max_steps',type=int,default=20000)
    p.add_argument('--weight_decay',type=float,default=0.1); p.add_argument('--grad_clip',type=float,default=1.0); p.add_argument('--dtype',default='bf16',choices=['bf16','fp16','fp32']); p.add_argument('--seed',type=int,default=1337)
    p.add_argument('--eval_interval',type=int,default=500); p.add_argument('--eval_iters',type=int,default=50); p.add_argument('--log_interval',type=int,default=20)
    p.add_argument('--grad_ckpt',action='store_true'); p.add_argument('--no_compile',dest='compile',action='store_false'); p.set_defaults(compile=True)
    p.add_argument('--infer_tokens',type=int,default=200,help='(compare) tokens to generate for the sample'); p.add_argument('--prompt',default='The history of')
    p.add_argument('--infer_scan',default='128,512,1024,4096,16384',help='(compare) comma lengths for inference throughput/VRAM/latency curves (16384 safe on 24GB: KV-cache ~0.77GB)')
    p.add_argument('--lerp_decode',default='auto',choices=['auto','graph','compile','eager'],help='(compare) model_lerp decode path: graph=CUDA graph (safe), compile=torch.compile reduce-overhead (Inductor, may crash ptxas->falls back), eager=python loop')
    p.add_argument('--no_plot',action='store_true',help='(compare) skip the comparison.png'); p.add_argument('--plot_out',default=None,help='(compare) output png path; default <out_dir>/comparison.png')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); a=p.parse_args()
    (train_compare if a.compare else train_anchor)(a)
if __name__=='__main__': main()
