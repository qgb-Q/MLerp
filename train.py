"""Industrial trainer for model_lerp (original 4-level six-projection skeleton + all upgrades). Pipeline:
read a tokenized .bin memmap (prebuilt by prepare_data.py from FineWeb, capped ~10GB; or auto-built from local HF-cache Arrow if present)
-> sample windows -> bf16 AdamW cosine+warmup, grad-accum/clip, optional grad-ckpt, periodic eval, ckpt save/resume,
CSV log + PNG loss curve. Single-GPU. torch.compile ON by default (--no_compile to disable).
  python prepare_data.py                                   # FIRST: stream FineWeb sample-10BT (stops at ~10GB text) -> out_lerp/{train,val}_neo.bin
  python train.py                                          # then train (finds the .bin, skips re-tokenizing)
  python train.py --n_layers 8 --batch_size 16 --max_steps 50000
  python train.py --think_mode lin --nest_mode mean        # architectural ablations from CLI
  python train.py --resume
NOTE: H=n_lobe*n_cortex must be divisible by nest_branch**nest_levels (=4 by default).
"""
import os,sys,glob,time,math,argparse,csv
import numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from model_lerp import LerpConfig,LerpLM
# ----------------------------- tokenizers -----------------------------
class ByteTok:
    name='byte'; vocab_size=256; eot=10                                  # newline as soft doc separator
    def encode(self,s): return list(s.encode('utf-8','ignore'))
    def decode(self,ids): return bytes(int(i)&255 for i in ids).decode('utf-8','replace')
class NeoTok:                                                            # GPT-2/GPT-Neo BPE (vocab 50257)
    name='neo'
    def __init__(self):
        import tiktoken; self.enc=tiktoken.get_encoding('gpt2'); self.vocab_size=self.enc.n_vocab; self.eot=self.enc.eot_token
    def encode(self,s): return self.enc.encode_ordinary(s)
    def decode(self,ids): return self.enc.decode([int(i) for i in ids])
def get_tok(name): return ByteTok() if name=='byte' else NeoTok()
# ----------------------------- data: arrow -> bin memmap -----------------------------
def find_arrow(data_dir,split):
    pats=[f'*{split}*.arrow']+(['*valid*.arrow'] if split=='validation' else [])
    fs=[]
    for p in pats: fs+=glob.glob(os.path.join(data_dir,p))
    return sorted(set(fs))
def load_split(data_dir,split):
    from datasets import Dataset,concatenate_datasets
    fs=find_arrow(data_dir,split)
    if not fs: return None
    print(f"[data] {split}: {len(fs)} arrow file(s) -> {[os.path.basename(f) for f in fs]}")
    return concatenate_datasets([Dataset.from_file(f) for f in fs])
def build_bin(data_dir,split,tok,out_path,dtype,text_col='text'):
    if os.path.exists(out_path): print(f"[data] {split}: cached {out_path} ({os.path.getsize(out_path)/1e6:.0f}MB)"); return os.path.getsize(out_path)//np.dtype(dtype).itemsize
    ds=load_split(data_dir,split)
    if ds is None: print(f"[data] {split}: NO arrow found in {data_dir}"); return 0
    if text_col not in ds.column_names: raise KeyError(f"column '{text_col}' not in {ds.column_names}")
    print(f"[data] {split}: tokenizing {len(ds)} docs -> {out_path}"); n=0; t0=time.time(); buf=[]
    with open(out_path,'wb') as f:
        for i in range(len(ds)):
            ids=tok.encode(ds[i][text_col]); ids.append(tok.eot); buf+=ids
            if len(buf)>=2_000_000: np.asarray(buf,dtype=dtype).tofile(f); n+=len(buf); buf=[]
            if i and i%200000==0: print(f"  {i}/{len(ds)} docs | {n/1e6:.1f}M tok | {time.time()-t0:.0f}s")
        if buf: np.asarray(buf,dtype=dtype).tofile(f); n+=len(buf)
    print(f"[data] {split}: {n/1e6:.2f}M tokens | {time.time()-t0:.0f}s"); return n
def make_batch(mm,n,B,T,device,pin):
    ix=np.random.randint(0,n-T-1,size=B,dtype=np.int64)                     # int64: corpus can exceed int32 max (2.1B); Windows numpy randint defaults to int32 -> overflow
    x=np.stack([mm[i:i+T] for i in ix]); y=np.stack([mm[i+1:i+1+T] for i in ix])
    x=torch.from_numpy(x.astype(np.int64)); y=torch.from_numpy(y.astype(np.int64))
    if pin: x=x.pin_memory(); y=y.pin_memory()
    return x.to(device,non_blocking=True),y.to(device,non_blocking=True)
# ----------------------------- train -----------------------------
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data_dir',default='./data',help="local HF-cache Arrow dir (FALLBACK only). Recommended: run prepare_data.py first to build out_dir/{train,val}_<tok>.bin (FineWeb-Edu); then this is unused.")
    p.add_argument('--out_dir',default='./out_lerp'); p.add_argument('--tokenizer',default='neo',choices=['byte','neo'])   # neo (GPT-2 BPE 50257) default: bigger vocab dilutes head saturation -> smaller loss-spike amplitude
    p.add_argument('--d_model',type=int,default=1024); p.add_argument('--d_cell',type=int,default=64)
    p.add_argument('--n_lobe',type=int,default=4); p.add_argument('--n_cortex',type=int,default=4)
    p.add_argument('--n_layers',type=int,default=12); p.add_argument('--chunk_len',type=int,default=256); p.add_argument('--ffn_mult',type=int,default=4)
    p.add_argument('--phi_expand',type=int,default=2,help='[#1] Q/K feature dim = phi_expand*d_cell (value stays d_cell)')
    p.add_argument('--no_qk_conv',dest='qk_conv',action='store_false'); p.set_defaults(qk_conv=True)   # [#2] short causal conv on Q/K
    p.add_argument('--think_mode',default='conv',choices=['conv','lin'],help='[#8] T_a self-think: conv(local) or lin(global)')
    p.add_argument('--nest_mode',default='tt',choices=['tt','mean'],help='[#4] cross-cortex aggregation: tt(learnable) or mean(fixed)')
    p.add_argument('--head_cap',type=float,default=5.0); p.add_argument('--head_eps',type=float,default=0.1); p.add_argument('--n_paths',type=int,default=64)   # [#3] FeynmanHead
    p.add_argument('--no_head_linear_base',dest='head_linear_base',action='store_false'); p.set_defaults(head_linear_base=True)   # [loss-floor fix] linear base removes large-vocab CE floor (~3.70 @neo). --no_head_linear_base = pure-Born ablation.
    p.add_argument('--use_delta',action='store_true')   # [recall/memory] DeltaNet erase-before-write in lin_read (recurrent=slower). Validate recall gain (ablation.py --exp delta) before relying on it.
    p.add_argument('--head_born',dest='head_born',action='store_true'); p.set_defaults(head_born=False)   # DEFAULT pure linear head (floor-free + drops ~2.4GB (B,L,V) born activations + 2 GEMMs @neo -> faster + more VRAM). --head_born re-enables Born (toy-only ~16% gain §10.5, no real-LM benefit).
    p.add_argument('--batch_size',type=int,default=2); p.add_argument('--block_size',type=int,default=2048); p.add_argument('--grad_accum',type=int,default=4)   # bs2*block2048*accum4 = 16384 eff_tok/step (printed at startup), lr 3e-4. NOTE: at block2048, B*L=4096 rows already saturates the GPU (compute-bound) -> raising batch does NOT raise tok/s (bs4 measured -> 16.1GB but SAME speed), it only grows VRAM+eff_tok. Real speed lever = MFU: torch.compile to fuse the fragmented scan kernels (reliable on Linux/WSL, fragile on native Windows ptxas), NOT batch. WINDOWS: exceeding VRAM spills to shared RAM over PCIe -> 10-50x slower; keep peak <24GB. BYTE (V=256): more headroom. Lean head: --no_head_born.
    p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--min_lr',type=float,default=3e-5); p.add_argument('--warmup',type=int,default=500)
    p.add_argument('--max_steps',type=int,default=20000); p.add_argument('--weight_decay',type=float,default=0.1); p.add_argument('--grad_clip',type=float,default=1.0)
    p.add_argument('--spike_skip',type=float,default=0.0,help='OPT-IN safety net (default OFF=0 so you can study raw dynamics): skip optimizer step if grad-norm > this * its EMA. Set ~4 only if you hit CATASTROPHIC divergence (loss stuck high many steps).')
    p.add_argument('--eval_interval',type=int,default=500); p.add_argument('--eval_iters',type=int,default=50); p.add_argument('--log_interval',type=int,default=20)
    p.add_argument('--save_interval',type=int,default=1000); p.add_argument('--grad_ckpt',action='store_true'); p.add_argument('--no_compile',dest='compile',action='store_false'); p.set_defaults(compile=True)   # compile default ON (your Windows+Triton works; --no_compile to disable)
    p.add_argument('--no_topup',dest='topup',action='store_false'); p.set_defaults(topup=True)   # DYNAMIC data: when training has drawn ~the whole corpus (step*eff_tok >= n_tr), auto-stream+APPEND --topup_gb more FineWeb (needs network + prepare_data's data_meta_<tok>.json). One-time stall per top-up. --no_topup disables.
    p.add_argument('--topup_gb',type=float,default=10.0,help='GB of fresh FineWeb text appended per auto-topup (raise for fewer/larger pulls -> cheaper resume)')
    p.add_argument('--dtype',default='bf16',choices=['bf16','fp16','fp32']); p.add_argument('--seed',type=int,default=1337); p.add_argument('--resume',action='store_true')
    p.add_argument('--init_from',default=None,help='start from this checkpoint (e.g. out_lerp/ckpt_best.pt); arch+tokenizer taken from it')
    p.add_argument('--reset_opt',action='store_true',help='with --init_from: fresh optimizer (recommended after a divergence)')
    p.add_argument('--reset_step',action='store_true',help='with --init_from: reset step to 0 and restart LR schedule')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); a=p.parse_args()
    os.makedirs(a.out_dir,exist_ok=True); torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev=a.device; cuda=dev.startswith('cuda')
    if cuda: torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    init_ck=None; saved_cfg=None; _ck=os.path.join(a.out_dir,'ckpt.pt')
    src=a.init_from or (_ck if (a.resume and os.path.exists(_ck)) else None)
    if src:
        if not os.path.exists(src): print(f"[fatal] checkpoint not found: {src}"); sys.exit(1)
        init_ck=torch.load(src,map_location=dev,weights_only=False); saved_cfg=dict(init_ck['cfg']); a.tokenizer=init_ck.get('tokenizer',a.tokenizer)   # arch+tokenizer FROM ckpt -> resume/init_from robust to the default tokenizer & arch flags
        print(f"[{'init_from' if a.init_from else 'resume'}] {src} | step={init_ck.get('step')} val={init_ck.get('best_val')} | arch+tokenizer({a.tokenizer}) taken from ckpt")
    elif a.resume: print("[resume] no checkpoint in out_dir -> starting fresh")
    tok=get_tok(a.tokenizer); dtype=np.uint8 if tok.vocab_size<=256 else np.uint16
    print(f"[cfg] tokenizer={tok.name} vocab={tok.vocab_size} device={dev} dtype={a.dtype} bin_dtype={np.dtype(dtype).name}")
    # data
    tr_bin=os.path.join(a.out_dir,f'train_{tok.name}.bin'); va_bin=os.path.join(a.out_dir,f'val_{tok.name}.bin')
    n_tr=build_bin(a.data_dir,'train',tok,tr_bin,dtype); build_bin(a.data_dir,'validation',tok,va_bin,dtype)
    if n_tr==0: print("[fatal] no training tokens: no prebuilt .bin in out_dir and no Arrow in --data_dir.\n        Run:  python prepare_data.py   (streams FineWeb-Edu -> out_lerp/train_neo.bin), then re-run train.py."); sys.exit(1)
    tr=np.memmap(tr_bin,dtype=dtype,mode='r'); n_tr=len(tr)
    if os.path.exists(va_bin) and os.path.getsize(va_bin)>0: va=np.memmap(va_bin,dtype=dtype,mode='r')
    else: cut=int(n_tr*0.99); va=tr[cut:]; tr=tr[:cut]; n_tr=len(tr); print(f"[data] no val arrow -> tail split: train={n_tr/1e6:.1f}M val={len(va)/1e6:.2f}M")
    n_va=len(va); print(f"[data] train={n_tr/1e6:.2f}M val={n_va/1e6:.2f}M tokens")
    splits={'train':(tr,n_tr),'val':(va,n_va)}
    tok_per_step=a.batch_size*a.block_size*a.grad_accum                          # tokens drawn per optimizer step -> consumed = step*tok_per_step (derived from step => resume-safe)
    meta_ok=os.path.exists(os.path.join(a.out_dir,f'data_meta_{tok.name}.json')); topup_on=a.topup and meta_ok
    if a.topup and not meta_ok: print(f"[topup] OFF: no data_meta_{tok.name}.json (re-run prepare_data.py to enable dynamic +{a.topup_gb:.0f}GB top-up).")
    elif topup_on: print(f"[topup] ON: will append +{a.topup_gb:.0f}GB FineWeb each time consumed reaches corpus size (~every {n_tr/tok_per_step:.0f} steps at current bs/accum).")
    # model
    cfg=LerpConfig(**saved_cfg) if saved_cfg else LerpConfig(vocab_size=tok.vocab_size,d_model=a.d_model,d_cell=a.d_cell,n_lobe=a.n_lobe,n_cortex=a.n_cortex,n_layers=a.n_layers,chunk_len=a.chunk_len,ffn_mult=a.ffn_mult,phi_expand=a.phi_expand,qk_conv=a.qk_conv,think_mode=a.think_mode,nest_mode=a.nest_mode,head_cap=a.head_cap,head_eps=a.head_eps,n_paths=a.n_paths,head_linear_base=a.head_linear_base,head_born=a.head_born,use_delta=a.use_delta)
    _g=cfg.nest_branch**cfg.nest_levels; assert (cfg.n_lobe*cfg.n_cortex)%_g==0,f"[fatal] H=n_lobe*n_cortex={cfg.n_lobe*cfg.n_cortex} must be divisible by nest_branch**nest_levels={_g}; adjust n_cortex"
    model=LerpLM(cfg).to(dev); print(f"[model] params={model.num_params()/1e6:.2f}M heads={cfg.n_lobe*cfg.n_cortex} dk={cfg.phi_expand*cfg.d_cell} layers={cfg.n_layers} think={cfg.think_mode} nest={cfg.nest_mode}")
    if a.compile: model=torch.compile(model); print("[model] torch.compile enabled")
    NOWD=('U_re','U_im','out_re','out_im')                                      # FeynmanHead: propagator U (near-identity Born phase) + complex vocab-embedding must NOT decay toward 0 (would erase Born suppression). name-match survives torch.compile's _orig_mod. prefix
    decay=[];nodecay=[]
    for n,pm in model.named_parameters(): (nodecay if pm.dim()<2 or any(k in n for k in NOWD) else decay).append(pm)
    try: opt=torch.optim.AdamW([{'params':decay,'weight_decay':a.weight_decay},{'params':nodecay,'weight_decay':0.0}],lr=a.lr,betas=(0.9,0.95),fused=cuda)
    except TypeError: opt=torch.optim.AdamW([{'params':decay,'weight_decay':a.weight_decay},{'params':nodecay,'weight_decay':0.0}],lr=a.lr,betas=(0.9,0.95))
    amp=(a.dtype!='fp32') and cuda; adt=torch.bfloat16 if a.dtype=='bf16' else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=(a.dtype=='fp16' and cuda))
    step=0; best=float('inf'); ckpt_path=os.path.join(a.out_dir,'ckpt.pt'); log_path=os.path.join(a.out_dir,'log.csv')
    if a.init_from and init_ck is not None:
        (model._orig_mod if hasattr(model,'_orig_mod') else model).load_state_dict(init_ck['model'])
        if not a.reset_opt and 'opt' in init_ck:
            try: opt.load_state_dict(init_ck['opt']); _o='+optimizer'
            except Exception as e: _o=f'(opt reload failed -> fresh: {e})'
        else: _o='(fresh optimizer)'
        if not a.reset_step: step=init_ck.get('step',0); best=init_ck.get('best_val',best)
        print(f"[init_from] weights loaded {_o}; "+(f"continue@step={step} best={best:.4f}" if not a.reset_step else "step=0, fresh LR schedule"))
    elif a.resume and init_ck is not None:
        (model._orig_mod if hasattr(model,'_orig_mod') else model).load_state_dict(init_ck['model']); opt.load_state_dict(init_ck['opt']); step=init_ck['step']; best=init_ck.get('best_val',best); print(f"[resume] step={step} best_val={best:.4f}")
    if not ((a.resume or (a.init_from and not a.reset_step)) and os.path.exists(log_path)):
        with open(log_path,'w',newline='') as f: csv.writer(f).writerow(['step','train_loss','val_loss','lr','tok_per_s','elapsed_s'])
    def lr_at(it):
        if it<a.warmup: return a.lr*(it+1)/a.warmup
        if it>=a.max_steps: return a.min_lr
        r=(it-a.warmup)/max(1,a.max_steps-a.warmup); return a.min_lr+0.5*(a.lr-a.min_lr)*(1+math.cos(math.pi*r))
    @torch.no_grad()
    def evaluate():
        model.eval(); out={}
        for sp,(mm,n) in splits.items():
            ls=[]
            for _ in range(a.eval_iters):
                x,y=make_batch(mm,n,a.batch_size,a.block_size,dev,cuda)
                with torch.autocast('cuda',dtype=adt,enabled=amp): _,l=model(x,y)
                ls.append(l.item())
            out[sp]=sum(ls)/len(ls)
        model.train(); return out
    print(f"[train] start step={step} max={a.max_steps} bs={a.batch_size} seq={a.block_size} accum={a.grad_accum} eff_tok/step={a.batch_size*a.block_size*a.grad_accum}")
    model.train(); t0=time.time(); tlog=t0; hist=[]; gn_ema=0.0; nskip=0; trace=[]; trace_path=os.path.join(a.out_dir,'loss_trace.csv')   # per-step (step,loss,gnorm) for loss_analysis.py spectral diagnosis
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
        spike=(a.spike_skip>0 and gn_ema>0 and gn>a.spike_skip*gn_ema)           # outlier-gradient batch -> drop the step (loss-spike guard)
        if spike: nskip+=1
        else: scaler.step(opt); scaler.update(); gn_ema=gn if gn_ema==0 else 0.97*gn_ema+0.03*gn
        step+=1; trace.append((step,lacc,gn))                                  # every-step signal for fluctuation analysis
        if topup_on and step*tok_per_step>=n_tr:                                # consumed ~= whole corpus -> pull fresh data instead of repeating
            print(f"[topup] consumed ~{step*tok_per_step/1e9:.2f}B tok >= corpus {n_tr/1e9:.2f}B -> streaming +{a.topup_gb:.0f}GB FineWeb & appending (one-time stall)...")
            try:
                from prepare_data import topup_bin                              # lazy import (avoids train<->prepare_data circular import at module load)
                added=topup_bin(a.out_dir,tok.name,a.topup_gb)
                if added>0:
                    tr=np.memmap(tr_bin,dtype=dtype,mode='r'); n_tr=len(tr); splits['train']=(tr,n_tr)   # reopen grown memmap; loop reads splits['train'] fresh
                    print(f"[topup] +{added/1e6:.1f}M tok -> corpus {n_tr/1e9:.2f}B ({os.path.getsize(tr_bin)/1e9:.2f}GB); next ~step {n_tr//tok_per_step}")
                else: topup_on=False; print("[topup] no meta -> disabled.")
            except Exception as e: topup_on=False; print(f"[topup] failed ({type(e).__name__}: {e}) -> disabled, continuing on existing data.")
        if step%a.log_interval==0:
            dt=time.time()-tlog; tps=a.batch_size*a.block_size*a.grad_accum*a.log_interval/max(dt,1e-9); tlog=time.time()
            print(f"step {step:6d} | loss {lacc:.4f} | gnorm {gn:6.2f} | lr {lr:.2e} | {tps:8.0f} tok/s | skip {nskip} | {time.time()-t0:6.0f}s")
            with open(trace_path,'w',newline='') as f: w=csv.writer(f); w.writerow(['step','loss','gnorm']); w.writerows([[s,f"{l:.4f}",f"{g:.3f}"] for s,l,g in trace])   # per-step trace, refreshed every log_interval -> python loss_analysis.py --csv out_lerp/loss_trace.csv
        if step%a.eval_interval==0 or step==a.max_steps:
            ev=evaluate(); tps=a.batch_size*a.block_size*a.grad_accum*a.log_interval/max(time.time()-tlog,1e-9)
            print(f"  [eval] step {step} train {ev['train']:.4f} val {ev['val']:.4f}"+("  *best*" if ev['val']<best else ""))
            with open(log_path,'a',newline='') as f: csv.writer(f).writerow([step,f"{ev['train']:.4f}",f"{ev['val']:.4f}",f"{lr:.3e}",f"{tps:.0f}",f"{time.time()-t0:.0f}"])
            hist.append((step,ev['train'],ev['val']))
            with open(trace_path,'w',newline='') as f: w=csv.writer(f); w.writerow(['step','loss','gnorm']); w.writerows([[s,f"{l:.4f}",f"{g:.3f}"] for s,l,g in trace])   # -> python loss_analysis.py --csv out_lerp/loss_trace.csv
            sd=(model._orig_mod if hasattr(model,'_orig_mod') else model).state_dict()
            ck={'model':sd,'opt':opt.state_dict(),'step':step,'cfg':cfg.__dict__,'tokenizer':tok.name,'best_val':min(best,ev['val'])}
            torch.save(ck,ckpt_path)
            if ev['val']<best: best=ev['val']; torch.save(ck,os.path.join(a.out_dir,'ckpt_best.pt'))
        elif step%a.save_interval==0:
            sd=(model._orig_mod if hasattr(model,'_orig_mod') else model).state_dict()
            torch.save({'model':sd,'opt':opt.state_dict(),'step':step,'cfg':cfg.__dict__,'tokenizer':tok.name,'best_val':best},ckpt_path)
    # loss curve
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        if hist:
            s,tr_,va_=zip(*hist); plt.figure(figsize=(8,5)); plt.plot(s,tr_,label='train'); plt.plot(s,va_,label='val')
            plt.xlabel('step'); plt.ylabel('loss'); plt.title(f'model_lerp ({tok.name}, {model.num_params()/1e6:.1f}M)'); plt.legend(); plt.grid(alpha=.3)
            plt.savefig(os.path.join(a.out_dir,'loss_curve.png'),dpi=120,bbox_inches='tight'); print(f"[done] curve -> {a.out_dir}/loss_curve.png")
    except Exception as e: print(f"[warn] curve skipped: {e}")
    print(f"[done] step={step} best_val={best:.4f} ckpt={ckpt_path}")
if __name__=='__main__': main()
