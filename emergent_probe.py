"""emergent_probe.py -- does model_lerp exhibit behavioral signatures a matched Transformer does NOT?
Controlled comparison on the TWO ALREADY-TRAINED ckpts (same data/budget/params). Each probe is grounded in a
SPECIFIC architectural difference and emits a falsifiable PREDICTION, the MEASURED number, and a VERDICT.
Probes: P1 length-extrapolation | P2 in-context induction & capacity | P3 state-trajectory (lerp-only observable)
        P4 predictive entropy & calibration | P5 repetition/attractor in free generation | P6 perturbation self-heal.
Usage:  python emergent_probe.py --lerp out_lerp/ckpt.pt --anchor out_anchor/ckpt.pt --bin_dir out_lerp
        python emergent_probe.py --selftest                 # tiny CPU smoke (byte tok, no ckpts/tiktoken needed)
NOTE: real numbers need the GPU + your real ckpts; --selftest only proves the interfaces/plumbing on CPU."""
import os,sys,math,argparse,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from contextlib import nullcontext
DEV='cuda' if torch.cuda.is_available() else 'cpu'
AMP=(lambda:torch.autocast('cuda',dtype=torch.bfloat16)) if DEV=='cuda' else (lambda:nullcontext())
def _val(bin_dir,tok):
    from anchoring_group import load_data
    _,_,va,n=load_data(bin_dir,tok)
    if va is None or n<8: print("[fatal] no val tokens"); sys.exit(1)
    return va,n
def _seq(va,n,L,rng): i=int(rng.integers(0,max(1,n-L-1))); return torch.from_numpy(np.asarray(va[i:i+L+1]).astype(np.int64))[None].to(DEV)
def _unigram(va,n,V,k=300000): c=np.bincount(np.asarray(va[:min(k,n)]).astype(np.int64),minlength=V).astype(np.float64); return c/max(c.sum(),1)
def _fwd(model,x,is_gpt,ctx):                                          # -> fp32 logits (1,T,V) or None if GPT asked beyond ctx
    if is_gpt and x.size(1)>ctx: return None
    with torch.no_grad(),AMP(): lg,_=model(x)
    return lg.float()
def _nll(model,x,is_gpt,ctx):                                         # per-position NLL of teacher-forced next token -> (T-1,) or None
    lg=_fwd(model,x[:,:-1],is_gpt,ctx)
    if lg is None: return None
    return (-F.log_softmax(lg,-1).gather(-1,x[:,1:,None]).squeeze(-1))[0]
# ----------------------------- P1: length extrapolation past training ctx -----------------------------
def p1_length(lerp,gpt,ctx,va,n,Lmax,nseq,rng,R):
    print("\n[P1] LENGTH EXTRAPOLATION  (trained at ctx=%d; can the recurrence generalize beyond?)"%ctx)
    print("     PREDICT: GPT per-pos NLL undefined past ctx (pos-emb cliff); model_lerp continues -- flat=true length-gen, rising=state saturates.")
    bins=np.unique(np.clip((np.geomspace(8,Lmax,24)).astype(int),8,Lmax)); acc_l={}; acc_g={}
    for s in range(nseq):
        x=_seq(va,n,Lmax,rng); nl=_nll(lerp,x,False,ctx); ng=_nll(gpt,x[:,:ctx+1],True,ctx)
        for b in bins:
            if b<nl.numel(): acc_l.setdefault(b,[]).append(nl[b].item())
            if ng is not None and b<ng.numel(): acc_g.setdefault(b,[]).append(ng[b].item())
    xs=sorted(acc_l); yl=[np.mean(acc_l[b]) for b in xs]; xg=sorted(acc_g); yg=[np.mean(acc_g[b]) for b in xg]
    inside=np.mean([v for b in xs if b<ctx for v in acc_l[b]]); outside=[np.mean(acc_l[b]) for b in xs if b>=ctx]
    R['p1']=dict(xs=xs,yl=yl,xg=xg,yg=yg,ctx=ctx)
    drift=(outside[-1]-inside) if outside else float('nan')
    print("     MEASURED: lerp NLL in-ctx=%.3f ; at %d=%.3f (drift %+.3f vs in-ctx) ; GPT stops at %d."%(inside,xs[-1],yl[-1],drift,ctx))
    print("     VERDICT : %s"%("model_lerp EXTRAPOLATES (NLL stays bounded past ctx) -- emergent length-gen the GPT cannot do." if drift<0.5 else "model_lerp degrades past ctx (state saturates), but still RUNS where GPT cannot."))
# ----------------------------- P2: in-context induction + capacity ceiling -----------------------------
def p2_induction(lerp,gpt,ctx,uni,ns,reps,rng,R):
    print("\n[P2] IN-CONTEXT INDUCTION  (feed [R;R]; 2nd-copy NLL drop = it learned the sequence in-context)")
    print("     PREDICT: both drop on copy-2 (induction); model_lerp's drop SHRINKS as n grows (fixed state can't hold long R); GPT holds within ctx.")
    curve_l=[]; curve_g=[]; perpos=None
    for n in ns:
        if 2*n>4096: continue
        dl=[]; dg=[]
        for _ in range(reps):
            Rk=rng.choice(len(uni),size=n,p=uni); x=torch.from_numpy(np.concatenate([Rk,Rk]).astype(np.int64))[None].to(DEV)
            nl=_nll(lerp,x,False,ctx); a=nl[2:n].mean().item(); b=nl[n+2:2*n-1].mean().item(); dl.append(a-b)
            if 2*n<=ctx:
                ng=_nll(gpt,x,True,ctx); dg.append((ng[2:n].mean()-ng[n+2:2*n-1].mean()).item())
            if n==ns[len(ns)//2] and perpos is None: perpos=(nl.cpu().numpy(), (None if 2*n>ctx else _nll(gpt,x,True,ctx).cpu().numpy()), n)
        curve_l.append((n,float(np.mean(dl)))); curve_g.append((n,float(np.mean(dg)) if dg else float('nan')))
    R['p2']=dict(curve_l=curve_l,curve_g=curve_g,perpos=perpos)
    nl0=curve_l[0][1]; nlN=curve_l[-1][1]
    print("     MEASURED: lerp induction-strength n=%d -> %.3f ; n=%d -> %.3f (%s)."%(curve_l[0][0],nl0,curve_l[-1][0],nlN,"falls off = capacity ceiling" if nlN<nl0-0.1 else "holds"))
    g_ok=[v for _,v in curve_g if not math.isnan(v)]
    print("     VERDICT : %s"%("model_lerp DOES form in-context recall, but it DECAYS with sequence length (state capacity) -- a Transformer holds flat within ctx." if (nlN<nl0-0.1) else "model_lerp's induction is roughly length-flat like the GPT within tested range."))
# ----------------------------- P3: state trajectory (UNIQUE to model_lerp) -----------------------------
def _state_traj(lerp,x):                                              # replicate generate()'s eager prefill+step, capture per-layer ||S1|| and ||dS1||
    emb=lerp.embed; T=x.size(1); h=emb(x[:,:1]); states=[]
    with torch.no_grad(),AMP():
        for ly in lerp.layers: h,st=ly(h,cap=True); states.append(st)
        nrm=[[states[i][2].float().norm().item() for i in range(len(states))]]; prev=[states[i][2].float() for i in range(len(states))]; dn=[[0.0]*len(states)]
        for t in range(1,T):
            h=emb(x[:,t:t+1])
            for i,ly in enumerate(lerp.layers): h,states[i]=ly.step(h,states[i])
            cur=[states[i][2].float() for i in range(len(states))]
            nrm.append([c.norm().item() for c in cur]); dn.append([(cur[i]-prev[i]).norm().item() for i in range(len(cur))]); prev=cur
    return np.array(nrm),np.array(dn)                                # (T,nlayer)
def p3_state(lerp,ctx,va,n,L,eot,rng,R):
    print("\n[P3] STATE TRAJECTORY  (model_lerp-ONLY: a Transformer has no single state vector to read)")
    print("     PREDICT: ||S|| grows then saturates; state WRITE ||dS|| spikes on high-surprise tokens (surprise-gated memory); responds at doc boundaries (eot).")
    x=_seq(va,n,L,rng); nrm,dn=_state_traj(lerp,x); nl=_nll(lerp,x,False,ctx).cpu().numpy()
    al=min(len(nl),dn.shape[0]-1); dwrite=dn[1:al+1].mean(1); surp=nl[:al]                      # dS1 writing token t vs surprise(token t)
    corr=float(np.corrcoef(dwrite,surp)[0,1]) if al>3 and dwrite.std()>0 and surp.std()>0 else float('nan')
    toks=x[0].cpu().numpy(); eot_pos=np.where(toks[:L]==eot)[0]
    last=nrm[-1].mean(); mid=nrm[len(nrm)//2].mean(); sat=(last-mid)/max(mid,1e-6)
    R['p3']=dict(nrm=nrm,dn=dn,surp=surp,dwrite=dwrite,eot_pos=eot_pos,corr=corr)
    print("     MEASURED: ||S|| mid=%.2f end=%.2f (late-growth %+.1f%%) ; corr(||dS||, surprise)=%.3f ; %d eot boundaries in window."%(mid,last,100*sat,corr,len(eot_pos)))
    print("     VERDICT : %s"%(("state WRITE correlates with surprise (r=%.2f) -- emergent surprise-gated memory, no Transformer analog."%corr) if (not math.isnan(corr) and corr>0.15) else "no strong surprise-write coupling; state mainly tracks accumulation."))
# ----------------------------- P4: predictive entropy + calibration -----------------------------
def _entcal(model,x,is_gpt,ctx,nb=15):
    lg=_fwd(model,x[:,:-1],is_gpt,ctx)
    if lg is None: return None
    p=F.softmax(lg,-1)[0]; tgt=x[0,1:]; ent=(-(p*torch.log(p+1e-12)).sum(-1)).cpu().numpy()
    conf,pred=p.max(-1); correct=(pred==tgt).float().cpu().numpy(); conf=conf.cpu().numpy()
    edg=np.linspace(0,1,nb+1); ece=0.0; rel=[]
    for j in range(nb):
        m=(conf>=edg[j])&(conf<edg[j+1])
        if m.sum()>0: a=correct[m].mean(); c=conf[m].mean(); ece+=(m.sum()/len(conf))*abs(a-c); rel.append((c,a))
    return ent,float(ece),np.array(rel),float(correct.mean())
def p4_entcal(lerp,gpt,ctx,va,n,L,nseq,rng,R):
    print("\n[P4] PREDICTIVE ENTROPY + CALIBRATION  (softmax-attn head vs FeynmanHead-on-integrated-state)")
    print("     PREDICT: integrated-state predictions may be SMOOTHER (higher entropy / less peaky); calibration (ECE) may differ.")
    EL=[]; EG=[]; el=eg=None; al=ag=None
    for _ in range(nseq):
        x=_seq(va,n,L,rng); rl=_entcal(lerp,x,False,ctx); rg=_entcal(gpt,x,True,ctx)
        if rl: EL.append(rl[0]); el=rl[1]; al=rl[3]; rel_l=rl[2]
        if rg: EG.append(rg[0]); eg=rg[1]; ag=rg[3]; rel_g=rg[2]
    EL=np.concatenate(EL); EG=np.concatenate(EG) if EG else None
    R['p4']=dict(EL=EL,EG=EG,ece_l=el,ece_g=eg,rel_l=rel_l,rel_g=(rel_g if EG is not None else None))
    print("     MEASURED: mean entropy lerp=%.3f gpt=%s ; ECE lerp=%.3f gpt=%s ; next-tok acc lerp=%.3f gpt=%s."%(EL.mean(),("%.3f"%EG.mean() if EG is not None else "n/a"),el,("%.3f"%eg if eg is not None else "n/a"),al,("%.3f"%ag if ag is not None else "n/a")))
    print("     VERDICT : %s"%(("model_lerp is %s-entropy and %s-calibrated vs GPT."%(("higher" if EG is not None and EL.mean()>EG.mean() else "lower"),("better" if eg is not None and el<eg else "worse"))) if EG is not None else "GPT side n/a (ctx)."))
# ----------------------------- P5: repetition / attractor in free generation -----------------------------
def _distinct(seq,k): g=[tuple(seq[i:i+k]) for i in range(len(seq)-k+1)]; return len(set(g))/max(1,len(g))
def _onset(seq,k=4):
    seen=set()
    for i in range(len(seq)-k+1):
        g=tuple(seq[i:i+k])
        if g in seen: return i
        seen.add(g)
    return len(seq)
def p5_repeat(lerp,gpt,ctx,va,n,plen,glen,temp,rng,R):
    print("\n[P5] REPETITION / ATTRACTOR  (greedy-ish free gen; fixed-state recurrence may converge to an attractor)")
    print("     PREDICT: model_lerp loops EARLIER / tighter (state fixed-point); GPT loops via self-attention to its own output.")
    plen=min(plen,max(2,ctx//2)); x=_seq(va,n,plen,rng)[:, :plen]
    with torch.no_grad(),AMP():
        gl=lerp.generate(x,glen,temperature=temp,top_k=50,fast='eager')[0,plen:].cpu().numpy().tolist()
        gg=gpt.generate(x,max(1,min(glen,ctx-plen-1)),temperature=temp,top_k=50)[0,plen:].cpu().numpy().tolist()
    dl={k:_distinct(gl,k) for k in (1,2,3)}; dg={k:_distinct(gg,k) for k in (1,2,3)}; ol=_onset(gl); og=_onset(gg)
    R['p5']=dict(dl=dl,dg=dg,ol=ol,og=og,nl=len(gl),ng=len(gg))
    print("     MEASURED: distinct-1/2/3 lerp=%.2f/%.2f/%.2f gpt=%.2f/%.2f/%.2f ; 4-gram loop onset lerp=%d gpt=%d tok."%(dl[1],dl[2],dl[3],dg[1],dg[2],dg[3],ol,og))
    print("     VERDICT : %s"%("model_lerp loops EARLIER (attractor-like) -- distinct lower / onset sooner." if (ol<og or dl[3]<dg[3]) else "model_lerp is no more repetitive than the GPT here."))
# ----------------------------- P6: perturbation self-heal -----------------------------
def p6_heal(lerp,gpt,ctx,va,n,L,corr_at,nseq,rng,R):
    print("\n[P6] PERTURBATION SELF-HEAL  (corrupt ONE token; how long does the damage persist downstream?)")
    print("     PREDICT: model_lerp's dNLL DECAYS with distance (corruption diluted out of the fixed state); GPT keeps it attendable within ctx.")
    L=min(L,ctx); dist=np.arange(1,L-corr_at-1); accl=[]; accg=[]
    for _ in range(nseq):
        x=_seq(va,n,L,rng); cl=_nll(lerp,x,False,ctx); cg=_nll(gpt,x,True,ctx)
        xp=x.clone(); xp[0,corr_at]=int(rng.integers(0,lerp.cfg.vocab_size)); pl=_nll(lerp,xp,False,ctx); pg=_nll(gpt,xp,True,ctx)
        accl.append((pl-cl).cpu().numpy()); accg.append((pg-cg).cpu().numpy())
    dl=np.mean(accl,0); dg=np.mean(accg,0); seg=lambda d:[float(np.mean(d[corr_at+1:corr_at+1+w])) for w in (8,64,256) if corr_at+1+w<=len(d)]
    sl=seg(dl); sg=seg(dg); R['p6']=dict(dl=dl,dg=dg,corr_at=corr_at)
    dec_l=(sl[0]-sl[-1]) if len(sl)>1 else float('nan'); dec_g=(sg[0]-sg[-1]) if len(sg)>1 else float('nan')
    print("     MEASURED: lerp dNLL @+8/+64/+256 = %s ; gpt = %s."%(["%.3f"%v for v in sl],["%.3f"%v for v in sg]))
    print("     VERDICT : %s"%("model_lerp SELF-HEALS faster (dNLL decays more with distance) -- fixed-state dilution forgets the bad token." if (dec_l>dec_g) else "GPT recovers at least as fast here."))
# ----------------------------- figure -----------------------------
def make_fig(R,out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(3,4,figsize=(22,13)); A=ax.ravel(); CL='#1f77b4'; CG='#d62728'
    for a in A: a.set_visible(False)
    if 'p1' in R:
        d=R['p1']; a=A[0]; a.set_visible(True); a.plot(d['xs'],d['yl'],'-o',ms=3,c=CL,label='model_lerp'); a.plot(d['xg'],d['yg'],'-o',ms=3,c=CG,label='anchor(GPT)')
        a.axvline(d['ctx'],ls='--',c='gray',lw=1); a.text(d['ctx'],a.get_ylim()[1],'train ctx',rotation=90,va='top',fontsize=7,color='gray')
        a.set_xscale('log'); a.set_xlabel('position'); a.set_ylabel('NLL (nats)'); a.set_title('(P1) length extrapolation\nGPT cliffs at ctx; lerp continues'); a.legend(fontsize=7)
    if 'p2' in R:
        d=R['p2']; a=A[1]; a.set_visible(True); xs=[n for n,_ in d['curve_l']]; a.plot(xs,[v for _,v in d['curve_l']],'-o',ms=3,c=CL,label='model_lerp'); a.plot([n for n,_ in d['curve_g']],[v for _,v in d['curve_g']],'-o',ms=3,c=CG,label='anchor(GPT)')
        a.set_xscale('log'); a.set_xlabel('sequence length n (to memorize)'); a.set_ylabel('induction strength (NLL drop)'); a.set_title('(P2) in-context recall vs capacity\nhigher=better recall'); a.legend(fontsize=7)
        if d['perpos']:
            b=A[2]; b.set_visible(True); pl,pg,nn=d['perpos']; b.plot(pl,c=CL,lw=.8,label='model_lerp'); 
            if pg is not None: b.plot(pg,c=CG,lw=.8,label='anchor(GPT)')
            b.axvline(nn,ls='--',c='gray',lw=1); b.set_xlabel('position in [R;R]'); b.set_ylabel('NLL'); b.set_title('(P2b) per-pos NLL over [R;R]\ndrop after | = induction'); b.legend(fontsize=7)
    if 'p4' in R:
        d=R['p4']; a=A[3]; a.set_visible(True); a.hist(d['EL'],bins=40,alpha=.6,color=CL,density=True,label='model_lerp')
        if d['EG'] is not None: a.hist(d['EG'],bins=40,alpha=.6,color=CG,density=True,label='anchor(GPT)')
        a.set_xlabel('predictive entropy (nats)'); a.set_ylabel('density'); a.set_title('(P4) next-token entropy\nright=less confident'); a.legend(fontsize=7)
    if 'p3' in R:
        d=R['p3']; a=A[4]; a.set_visible(True); nrm=d['nrm']
        for i in range(nrm.shape[1]): a.plot(nrm[:,i],lw=.7,alpha=.7)
        for e in d['eot_pos']: a.axvline(e,ls=':',c='k',lw=.5,alpha=.5)
        a.set_xlabel('position'); a.set_ylabel('||S|| per layer'); a.set_title('(P3) STATE NORM trajectory [lerp-only]\ndotted=eot doc boundary'); 
        b=A[5]; b.set_visible(True); b.scatter(d['surp'],d['dwrite'],s=4,alpha=.4,c=CL); b.set_xlabel('token surprise NLL(t)'); b.set_ylabel('state write ||dS||(t)'); b.set_title('(P3b) surprise-gated write\nr=%.2f'%d['corr'])
    if 'p4' in R and R['p4']['rel_l'] is not None:
        d=R['p4']; a=A[6]; a.set_visible(True); a.plot([0,1],[0,1],ls='--',c='gray',lw=1); rl=d['rel_l']; a.plot(rl[:,0],rl[:,1],'-o',ms=3,c=CL,label='lerp ECE %.3f'%d['ece_l'])
        if d['rel_g'] is not None: rg=d['rel_g']; a.plot(rg[:,0],rg[:,1],'-o',ms=3,c=CG,label='gpt ECE %.3f'%d['ece_g'])
        a.set_xlabel('confidence'); a.set_ylabel('empirical accuracy'); a.set_title('(P4b) reliability / calibration'); a.legend(fontsize=7)
    if 'p5' in R:
        d=R['p5']; a=A[7]; a.set_visible(True); ks=[1,2,3]; w=.35; xp=np.arange(3)
        a.bar(xp-w/2,[d['dl'][k] for k in ks],w,color=CL,label='model_lerp'); a.bar(xp+w/2,[d['dg'][k] for k in ks],w,color=CG,label='anchor(GPT)')
        a.set_xticks(xp); a.set_xticklabels(['distinct-1','distinct-2','distinct-3']); a.set_ylabel('fraction unique'); a.set_title('(P5) generation diversity\nonset lerp=%d gpt=%d'%(d['ol'],d['og'])); a.legend(fontsize=7)
    if 'p6' in R:
        d=R['p6']; a=A[8]; a.set_visible(True); c=d['corr_at']; xl=np.arange(len(d['dl']))-c; a.plot(xl,d['dl'],c=CL,lw=1,label='model_lerp'); a.plot(np.arange(len(d['dg']))-c,d['dg'],c=CG,lw=1,label='anchor(GPT)')
        a.axvline(0,ls='--',c='gray',lw=1); a.set_xlim(0,min(300,len(d['dl'])-c)); a.set_xlabel('tokens after corruption'); a.set_ylabel('dNLL (corrupt - clean)'); a.set_title('(P6) perturbation self-heal\ndecay=forgets bad token'); a.legend(fontsize=7)
    fig.suptitle('model_lerp vs anchor(GPT) -- emergent behavioral probes (controlled, same data/budget)',fontsize=14); fig.tight_layout(rect=[0,0,1,.98]); fig.savefig(out,dpi=110); print("\n[fig] wrote %s"%out)
# ----------------------------- selftest: tiny CPU models + data (byte tok, no tiktoken/ckpts) -----------------------------
def _selftest():
    from model_lerp import LerpConfig,LerpLM; from anchoring_group import GPT; from dataclasses import asdict
    import tempfile; d=tempfile.mkdtemp(); V=256
    cfg=LerpConfig(vocab_size=V,d_model=64,d_cell=16,n_lobe=2,n_cortex=2,n_layers=2,chunk_len=16,n_paths=8); lm=LerpLM(cfg)
    torch.save({'model':lm.state_dict(),'cfg':asdict(cfg)},os.path.join(d,'lerp.pt'))
    g=GPT(V,64,4,2,128,32,tie=True); torch.save({'model':g.state_dict(),'cfg':dict(vocab=V,d=64,nh=4,L=2,dff=128,ctx=32,tie=True)},os.path.join(d,'gpt.pt'))
    va=(np.random.rand(20000)*V).astype(np.uint8); va[::97]=10; va.tofile(os.path.join(d,'val_byte.bin')); np.array([],dtype=np.uint8).tofile(os.path.join(d,'train_byte.bin'))
    open(os.path.join(d,'train_byte.bin'),'wb').write((np.random.rand(2000)*V).astype(np.uint8).tobytes())
    return d
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--lerp',default='out_lerp/ckpt.pt'); p.add_argument('--anchor',default='out_anchor/ckpt.pt'); p.add_argument('--bin_dir',default='out_lerp')
    p.add_argument('--tokenizer',default='neo'); p.add_argument('--out',default='emergent.png'); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--probes',default='all',help='comma subset of 1,2,3,4,5,6 or all'); p.add_argument('--selftest',action='store_true')
    p.add_argument('--Lmax',type=int,default=8192); p.add_argument('--nseq',type=int,default=6); p.add_argument('--gen',type=int,default=400)
    a=p.parse_args()
    if a.selftest: a.bin_dir=_selftest(); a.tokenizer='byte'; a.lerp=os.path.join(a.bin_dir,'lerp.pt'); a.anchor=os.path.join(a.bin_dir,'gpt.pt'); a.Lmax=64; a.nseq=2; a.gen=40
    from anchoring_group import load_lerp,load_anchor; from train import get_tok
    tok=get_tok(a.tokenizer); lerp=load_lerp(a.lerp,DEV); gpt=load_anchor(a.anchor,DEV); ctx=gpt.ctx; va,n=_val(a.bin_dir,tok); uni=_unigram(va,n,tok.vocab_size); rng=np.random.default_rng(a.seed)
    print("[setup] dev=%s lerp=%.1fM gpt=%.1fM ctx=%d val=%.2fM tok eot=%d"%(DEV,lerp.num_params()/1e6,gpt.num_params()/1e6,ctx,n/1e6,tok.eot))
    sel=set(range(1,7)) if a.probes=='all' else set(int(i) for i in a.probes.split(',')); R={}
    if a.selftest: ns2=[16,32]; reps=2; LL=24
    else: ns2=[32,64,128,256,512,1024]; reps=8; LL=min(2048,ctx)
    if 1 in sel: p1_length(lerp,gpt,ctx,va,n,a.Lmax,a.nseq,rng,R)
    if 2 in sel: p2_induction(lerp,gpt,ctx,uni,ns2,reps,rng,R)
    if 3 in sel: p3_state(lerp,ctx,va,n,LL,tok.eot,rng,R)
    if 4 in sel: p4_entcal(lerp,gpt,ctx,va,n,LL,a.nseq,rng,R)
    if 5 in sel: p5_repeat(lerp,gpt,ctx,va,n,32,a.gen,0.8,rng,R)
    if 6 in sel: p6_heal(lerp,gpt,ctx,va,n,LL,(4 if a.selftest else 16),a.nseq,rng,R)
    try: make_fig(R,a.out)
    except Exception as e: print("[fig] skipped (%s: %s)"%(type(e).__name__,e))
    print("\n[done] probes run: %s"%sorted(sel))
if __name__=='__main__': main()
