"""model_lerp: original 4-level brain skeleton (cell->cortex->lobe->whole) + all upgrades.
ONE primitive: I=Query, T=Key, C=Value read against a LINEAR matrix ledger (causal linear attention)
-> de-KV + fully parallel + causal. Hierarchy = grouping of linear-attention HEADS (cortices); cross-cortex
aggregation = learnable HierAgg (TT-RNN core, replaces mean-pool). Output -> RMSNorm+FFN -> FeynmanHead.

WHY IT NEVER EXPLODES: lin_read is CHUNKED -- it carries a (dk,dv) state across chunks, NO L dimension in
the ledger. State is O(B*H*dk*dv), NOT O(B*H*L*dk*dv). 256 cortices * [128,64] ledger is tiny; the de-KV
trick removes the sequence dimension entirely. Mathematically identical to the cumsum reference (verified T0).

==================================== ARCHITECTURE (per layer) ====================================
four projections: to_Ix (token query) + to_Ts/Cs/Is (whole-brain key/value/self-think). [lerp gate + hop2 REMOVED -> to_Tx/to_Cx dropped]
 read           : Xl = lin_read(Ix ; brain Ts,Cs)                      # token reads brain memory (the ONE scan)
 self-think     : Ta = conv(Is)  (think_mode='conv', local recombination) | lin_read(Is;Ts,Cs) ablation
 aggregate      : Ta -> HierAgg (learnable cross-cortex, init=mean)     # cell->cortex->lobe->whole reduce/restore
 assemble       : feat=[Xl ; HierAgg(Ta)] -> proj_o -> +X residual -> +FFN
==================================== UPGRADES =====================================================
1 phi-expand  dk=phi_expand*dc on Q/K (value stays dc) -> separable keys -> better RECALL
2 qk_conv     short causal depthwise conv on Ix,Ts before phi -> local selectivity
3 FeynmanHead complex Born + PxP path-integral propagator + head_eps (anti grad-cliff) + head_cap
4 HierAgg     learnable TT-RNN cross-cortex core replacing fixed mean-pool (init=mean, residual)
5 lin_read    chunked, state O(B*H*dk*dv) -> never explodes
6 forget-gate DELETED (recency bias destroys associative recall 0.96->0.06)
7 single-read lerp gate + hop2 REMOVED (hop2 redundant: reads same ledger as Xl, -hop2 ablation neutral §10[2]; gate's only consumer was hop2) -> 1 scan/layer
8 T_a -> conv causal depthwise temporal conv (local self-think) replaces global lin self-read
=================================================================================================="""
import math,torch,torch.nn as nn,torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint
from dataclasses import dataclass
def _lin_read_delta_ref(I,T,C,beta,eps=1e-6):                                            # SEQUENTIAL DeltaNet reference (slow O(L)) -- ground truth for testing the chunked delta branch in lin_read
    B,H,L,dk=I.shape; dv=C.shape[-1]; qi=F.elu(I)+1; kt=F.elu(T)+1
    if not torch.is_tensor(beta): beta=torch.as_tensor(beta,device=I.device,dtype=qi.dtype)
    ph=(beta.dim()==1); bh=beta.view(1,H,1).to(qi.dtype) if ph else beta
    qi=qi/qi.norm(dim=-1,keepdim=True).clamp_min(eps); kt=kt/kt.norm(dim=-1,keepdim=True).clamp_min(eps)
    S=torch.zeros(B,H,dk,dv,device=I.device,dtype=qi.dtype); z=torch.zeros(B,H,dk,device=I.device,dtype=qi.dtype); out=torch.empty(B,H,L,dv,device=I.device,dtype=qi.dtype)
    for t in range(L):
        q,k,v=qi[:,:,t],kt[:,:,t],C[:,:,t]; b=bh if ph else bh[:,:,t]
        v_old=(k.unsqueeze(-2)@S).squeeze(-2); S=S+b.unsqueeze(-1)*(k.unsqueeze(-1)*(v-v_old).unsqueeze(-2)); z=z+b*k
        num=(q.unsqueeze(-2)@S).squeeze(-2); den=(q*z).sum(-1,keepdim=True).clamp_min(eps); out[:,:,t]=num/den
    return out
def lin_read(I,T,C,chunk=64,eps=1e-6,beta=None,ret_state=False):
    # de-KV linear attention. beta=None -> ADDITIVE chunked (S+=k^Tv). beta!=None -> DELTA RULE (DeltaNet): erase-before-write
    #   S += beta * phi(k)^T (v - phi(k)@S)  -> removes the value currently stored at key k before writing v, killing the
    #   additive-interference that caps recall (SOTA associative memory; DeltaNet=perfect MQAR). beta: write-gate, scalar/(H,)/(B,H,L,1).
    #   CHUNKED-PARALLEL (WY/Householder form, Yang et al. 2024): O(L/chunk) sequential, each chunk fully parallel -> ~chunk x fewer
    #   sequential steps than per-token. Verified == _lin_read_delta_ref (sequential) + gradcheck on CPU.
    B,H,L,dk=I.shape; dv=C.shape[-1]; qi=F.elu(I)+1; kt=F.elu(T)+1
    if beta is not None:
        if not torch.is_tensor(beta): beta=torch.as_tensor(beta,device=I.device,dtype=qi.dtype)
        wd=qi.dtype; cd=torch.float32 if wd in (torch.float16,torch.bfloat16) else wd    # compute in fp32 ONLY for low-precision (recurrence/inverse stability); keep fp32/fp64 native so autograd/gradcheck are exact
        qi=qi.to(cd); kt=kt.to(cd); Cf=C.to(cd)
        qi=qi/qi.norm(dim=-1,keepdim=True).clamp_min(eps); kt=kt/kt.norm(dim=-1,keepdim=True).clamp_min(eps)   # UNIT-NORM q,k: erase (I-beta k k^T) is a stable projection (eig 1-beta in [0,1)); with unit keys query=key retrieves the stored value
        bf=(beta.view(1,H,1).expand(B,H,L) if beta.dim()==1 else beta.squeeze(-1)).to(cd)   # (B,H,L) per-position write-gate
        cl=min(chunk,L); pad=(cl-L%cl)%cl
        if pad: qi=F.pad(qi,(0,0,0,pad)); kt=F.pad(kt,(0,0,0,pad)); Cf=F.pad(Cf,(0,0,0,pad)); bf=F.pad(bf,(0,pad))   # pad tokens get beta=0 -> no write
        Lp=L+pad; nc=Lp//cl; eye=torch.eye(cl,device=I.device,dtype=cd)
        qc=qi.view(B,H,nc,cl,dk); kc=kt.view(B,H,nc,cl,dk); vc=Cf.view(B,H,nc,cl,dv); bc=bf.view(B,H,nc,cl)
        S=torch.zeros(B,H,dk,dv,device=I.device,dtype=cd); z=torch.zeros(B,H,dk,device=I.device,dtype=cd); out=torch.empty(B,H,nc,cl,dv,device=I.device,dtype=cd)
        for c in range(nc):
            q,k,v,b=qc[:,:,c],kc[:,:,c],vc[:,:,c],bc[:,:,c]                               # (B,H,cl,*) ; b (B,H,cl)
            M=eye+torch.tril(b.unsqueeze(-1)*(k@k.transpose(-1,-2)),-1)                    # I + strict-lower(diag(b) K K^T): unit lower-tri
            U=torch.linalg.solve_triangular(M,b.unsqueeze(-1)*(v-k@S),upper=False,unitriangular=True)   # M^{-1} diag(b)(V-K S0): WY 'new values' (u_t=b_t(v_t-k_t S_{t-1}))
            qk=torch.tril(q@k.transpose(-1,-2),0)                                         # read-AFTER-write (incl diagonal)
            num=q@S+qk@U                                                                  # q S0 + intra
            den=((q*z.unsqueeze(2)).sum(-1)+(qk@b.unsqueeze(-1)).squeeze(-1)).clamp_min(eps)   # q z0 + intra occupancy
            out[:,:,c]=num/den.unsqueeze(-1); S=S+k.transpose(-1,-2)@U; z=z+(b.unsqueeze(-1)*k).sum(2)   # write chunk out + carry state/occupancy
        o=out.reshape(B,H,Lp,dv)[:,:,:L].to(wd); return (o,S,z) if ret_state else o          # S,z (fp32) carry the recurrent memory -> incremental decode
    pad=(chunk-L%chunk)%chunk
    if pad: qi=F.pad(qi,(0,0,0,pad)); kt=F.pad(kt,(0,0,0,pad)); C=F.pad(C,(0,0,0,pad))
    nc=(L+pad)//chunk
    qi=qi.view(B,H,nc,chunk,dk); kt=kt.view(B,H,nc,chunk,dk); Cc=C.view(B,H,nc,chunk,dv)
    S=torch.zeros(B,H,dk,dv,device=I.device,dtype=qi.dtype); z=torch.zeros(B,H,dk,device=I.device,dtype=qi.dtype)
    mask=torch.tril(torch.ones(chunk,chunk,device=I.device,dtype=qi.dtype)); out=torch.empty(B,H,nc,chunk,dv,device=I.device,dtype=qi.dtype)  # pre-alloc (compile-friendly: avoids list.append+cat graph-break, static-unrollable loop)
    for c in range(nc):
        q,k,v=qi[:,:,c],kt[:,:,c],Cc[:,:,c]                                  # (B,H,chunk,*)
        att=(q@k.transpose(-1,-2))*mask                                      # causal intra-chunk scores
        num=q@S+att@v                                                        # inter(state) + intra numerator
        den=((q*z.unsqueeze(2)).sum(-1,keepdim=True)+att.sum(-1,keepdim=True)).clamp_min(eps)
        out[:,:,c]=num/den
        S=S+k.transpose(-1,-2)@v; z=z+k.sum(2)                              # carry (dk,dv) state to next chunk
    o=out.reshape(B,H,nc*chunk,dv)[:,:,:L]; return (o,S.float(),z.float()) if ret_state else o   # final (S,z) carry the recurrent memory -> incremental decode (fp32 for step stability)
def lin_read_step(I,T,C,S,z,beta=None,eps=1e-6):
    # O(1) SINGLE-TOKEN update for incremental decoding. I,T:(B,H,1,dk) C:(B,H,1,dv) ; S:(B,H,dk,dv) z:(B,H,dk) fp32. returns out:(B,H,1,dv), S,z.
    # MUST match lin_read's per-position math (read-AFTER-write). Verified == full lin_read in the self-test.
    wd=I.dtype; q=(F.elu(I)+1).squeeze(2).float(); k=(F.elu(T)+1).squeeze(2).float(); v=C.squeeze(2).float()   # (B,H,*) fp32
    if beta is not None:
        if not torch.is_tensor(beta): beta=torch.as_tensor(beta,device=I.device,dtype=torch.float32)
        b=(beta.view(1,-1) if beta.dim()==1 else beta.reshape(q.shape[0],-1)).float().unsqueeze(-1)   # (1orB,H,1)
        q=q/q.norm(dim=-1,keepdim=True).clamp_min(eps); k=k/k.norm(dim=-1,keepdim=True).clamp_min(eps)         # unit-norm (delta stability)
        v_old=(k.unsqueeze(-2)@S).squeeze(-2); S=S+b.unsqueeze(-1)*(k.unsqueeze(-1)*(v-v_old).unsqueeze(-2)); z=z+b*k   # erase-before-write
    else:
        qk=(q*k).sum(-1,keepdim=True)                                                                 # (B,H,1) self (diagonal)
        S=S+k.unsqueeze(-1)*v.unsqueeze(-2); z=z+k                                                     # additive write (then read S incl this token)
    num=(q.unsqueeze(-2)@S).squeeze(-2); den=(q*z).sum(-1,keepdim=True).clamp_min(eps)                 # read AFTER write
    return (num/den).unsqueeze(2).to(wd),S,z
def _lin_read_ref(I,T,C,eps=1e-6):
    # cumsum reference (memory-heavy) -- ONLY to verify lin_read equivalence.
    qi=F.elu(I)+1; kt=F.elu(T)+1; kv=(kt.unsqueeze(-1)*C.unsqueeze(-2)).cumsum(2); z=kt.cumsum(2)
    return (qi.unsqueeze(-1)*kv).sum(-2)/(qi*z).sum(-1,keepdim=True).clamp_min(eps)
@dataclass
class LerpConfig:
    vocab_size:int=256
    d_model:int=512                       # residual width (user spec)
    d_cell:int=64                         # dc: value dim per cortex. dk=phi_expand*dc=128 (key/query). "fill dc to task-sufficiency, then add cortices"
    n_lobe:int=4                          # user spec: 4 lobes
    n_cortex:int=64                       # user spec: 64 cortices/lobe -> H=256 cortices(heads). NOTE param cost ~ d_model*H*dk; shrink n_cortex for small runs. H MUST be divisible by nest_branch**nest_levels.
    n_layers:int=4
    chunk_len:int=256                     # de-KV scan block. PURE engineering knob (math-invariant, verified fp32 7.2e-7). bigger chunk -> larger/fewer scan matmuls -> better GPU util at ctx>=512, same output.
    ffn_mult:int=4
    dropout:float=0.0
    tie:bool=True
    phi_expand:int=2                      # [#1] key/query feature dim = phi_expand*d_cell. value stays d_cell. bigger -> separable keys -> better RECALL.
    qk_conv:bool=True                     # [#2] short causal depthwise conv on Ix,Ts before phi (local selectivity, standard linear-attn position)
    qk_kernel:int=4                       # Q/K short-conv kernel
    think_mode:str='conv'                 # [#8] T_a self-think: 'conv'=causal depthwise temporal conv (local); 'lin'=lin_read self-read (global, ablation)
    think_kernel:int=4                    # causal depthwise conv kernel for think_mode='conv'
    nest_mode:str='tt'                    # [#4] cross-cortex aggregation: 'tt'=learnable TT-RNN core (init=mean, +recall); 'mean'=fixed mean-pool (legacy)
    nest_branch:int=2                     # hierarchy branching (group `branch` cortices per parent per level)
    nest_levels:int=2                     # nesting depth (cell->cortex->lobe->whole). H must be divisible by branch**levels
    head_eps:float=0.1                    # [#3] in-log floor: bounds Born grad amplifier to 1/sqrt(eps) (1e-6->1000x cliff; 0.1->3.2x)
    head_cap:float=5.0                    # [#3] Born-correction range +-cap (logits). 5=stable; 30=saturates small vocab.
    n_paths:int=64                        # [#3] Feynman path count P (complex propagator U is PxP)
    head_linear_base:bool=True            # [loss-floor fix] add unbounded linear base to FeynmanHead -> removes large-vocab CE floor (pure-Born caps CE>=~3.70 nats @V=50257). False=pure-Born ablation. Tied to embedding when cfg.tie.
    use_delta:bool=False                  # [recall/memory upgrade] DeltaNet erase-before-write in lin_read (S+=beta*k^T(v-k@S)) -> kills additive interference, SOTA associative recall. RECURRENT (slower); validate on MQAR/recall before defaulting on. False=additive (verified).
    head_born:bool=False                  # DEFAULT pure linear head (floor-free, leanest): drops born vr,vi (B,L,V) activations (~2.4GB @neo) + 2 (B,L,V) GEMMs -> faster + more VRAM. Born benefit only on a toy forbidden-mass task (~16%, §10.5), zero real-LM gain. head_born=True re-enables the Born path.
class RMSNorm(nn.Module):
    def __init__(self,d,eps=1e-6): super().__init__(); self.w=nn.Parameter(torch.ones(d)); self.eps=eps
    def forward(self,x): return x*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+self.eps).to(x.dtype)*self.w
class HierAgg(nn.Module):
    # [#4] learnable cross-cortex (cross-head) hierarchical aggregation. cell->cortex->lobe->whole reduce, restore broadcast, residual.
    # TT-RNN style: per level group `branch` siblings, mix by learnable weights (init uniform=mean), transform by learnable matrix (init identity).
    # init reduces EXACTLY to mean-pool; learns to do better. mode='mean' = fixed legacy pool.
    def __init__(self,H,dc,branch=2,levels=2,mode='tt'):
        super().__init__(); self.H=H; self.dc=dc; self.branch=branch; self.levels=levels; self.mode=mode
        assert H%(branch**levels)==0,f"H={H} must be divisible by branch**levels={branch**levels}"
        if mode=='tt':
            self.mix=nn.ParameterList([nn.Parameter(torch.zeros(branch)) for _ in range(levels)])  # softmax(0)=uniform=mean at init
            self.tf=nn.ModuleList([nn.Linear(dc,dc,bias=False) for _ in range(levels)])
            for lin in self.tf: nn.init.eye_(lin.weight)                                            # identity at init
    def forward(self,z):                                                    # (B,H,L,dc) -> z + whole-brain context (broadcast)
        B,H,L,dc=z.shape; g=self.branch**self.levels
        if self.mode=='mean':
            zz=z.view(B,H//g,g,L,dc); ctx=zz.mean(2,keepdim=True).expand_as(zz).reshape(B,H,L,dc); return z+ctx
        ctx=z                                                               # progressive up-reduce
        for lvl in range(self.levels):
            br=self.branch; Hc=ctx.shape[1]
            gg=ctx.view(B,Hc//br,br,L,dc); w=torch.softmax(self.mix[lvl],0)
            ctx=self.tf[lvl]((gg*w.view(1,1,br,1,1)).sum(2))                # weighted-aggregate siblings + transform
        ctx=ctx.view(B,H//g,1,L,dc).expand(B,H//g,g,L,dc).reshape(B,H,L,dc) # restore: broadcast whole-brain summary back
        return z+ctx
class LerpLayer(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg; d=cfg.d_model; H=cfg.n_lobe*cfg.n_cortex; dc=cfg.d_cell; dk=cfg.phi_expand*dc
        self.H=H; self.dc=dc; self.dk=dk; self.ch=cfg.chunk_len; self.think_mode=cfg.think_mode
        # four projections: Ix token query (dk) ; whole-brain Ts key (dk), Cs value (dc), Is self-think. lerp gate + hop2 REMOVED -> to_Tx/to_Cx (fed only the gate) + Wil/Wtl/Wcl/a_* dropped.
        self.to_Ix=nn.Linear(d,H*dk,bias=False)
        self.to_Ts=nn.Linear(d,H*dk,bias=False); self.to_Cs=nn.Linear(d,H*dc,bias=False)
        self.to_Is=nn.Linear(d,H*(dc if cfg.think_mode=='conv' else dk),bias=False)   # conv-Ta needs dc; lin-Ta needs dk (query Ts ledger)
        if cfg.use_delta: self.delta_beta=nn.Parameter(torch.full((H,),2.0))   # [recall] per-head delta write-gate; beta=sigmoid(.)~0.88 init. created only when use_delta (no dead param when off)
        # [#2] qk short conv (depthwise per (head,feature)) on Ix and Ts
        if cfg.qk_conv:
            self.conv_q=nn.Conv1d(H*dk,H*dk,cfg.qk_kernel,groups=H*dk,bias=False)
            self.conv_k=nn.Conv1d(H*dk,H*dk,cfg.qk_kernel,groups=H*dk,bias=False)
        # [#8] self-think temporal conv (depthwise per (head,cell))
        if cfg.think_mode=='conv': self.think_conv=nn.Conv1d(H*dc,H*dc,cfg.think_kernel,groups=H*dc,bias=False)
        # [#4] learnable cross-cortex aggregation for the self-think read (agg_r dropped with hop2)
        self.agg_t=HierAgg(H,dc,cfg.nest_branch,cfg.nest_levels,cfg.nest_mode)
        self.ln_in=RMSNorm(d); self.ln_ff=RMSNorm(d); self.proj_o=nn.Linear(2*H*dc,d,bias=False)
        self.ffn=nn.Sequential(nn.Linear(d,cfg.ffn_mult*d),nn.GELU(),nn.Linear(cfg.ffn_mult*d,d))
    def _h(self,x,dd): B,L,_=x.shape; return x.view(B,L,self.H,dd).permute(0,2,1,3)   # (B,L,H*dd)->(B,H,L,dd)
    def _dwconv(self,z,conv,k,cap=False):                                   # causal depthwise conv over time. z:(B,H,L,dd). cap -> also return cache (last k-1 raw inputs) for incremental decode
        B,Hh,L,dd=z.shape; zz=z.permute(0,1,3,2).reshape(B,Hh*dd,L)
        cache=None
        if cap and k>1: cache=zz[:,:,-(k-1):]; cache=F.pad(cache,(k-1-cache.shape[-1],0)) if cache.shape[-1]<k-1 else cache
        o=conv(F.pad(zz,(k-1,0))).reshape(B,Hh,dd,L).permute(0,1,3,2)
        return (o,cache) if cap else o
    def _dwconv_step(self,z,conv,k,cache):                                  # single-token causal conv via cache. z:(B,H,1,dd), cache:(B,Hh*dd,k-1)
        B,Hh,_,dd=z.shape; zt=z.permute(0,1,3,2).reshape(B,Hh*dd,1)
        if k==1: return conv(zt).reshape(B,Hh,dd,1).permute(0,1,3,2),cache
        win=torch.cat([cache,zt],-1); o=conv(win).reshape(B,Hh,dd,1).permute(0,1,3,2); return o,win[:,:,1:]
    def forward(self,X,cap=False):                                          # cap=True -> also capture recurrent state (lin_read S,z + conv caches) for incremental decode prefill. cap=False path is byte-identical to training.
        B,L,d=X.shape; h=self.ln_in(X); ch=self.ch; dk=self.dk; cq=ck=None
        Ix=self._h(self.to_Ix(h),dk); Ts=self._h(self.to_Ts(h),dk); Cs=self._h(self.to_Cs(h),self.dc)
        Is=self._h(self.to_Is(h),self.dc if self.think_mode=='conv' else dk)
        if self.cfg.qk_conv:
            if cap: Ix,cq=self._dwconv(Ix,self.conv_q,self.cfg.qk_kernel,cap=True); Ts,ck=self._dwconv(Ts,self.conv_k,self.cfg.qk_kernel,cap=True)
            else: Ix=self._dwconv(Ix,self.conv_q,self.cfg.qk_kernel); Ts=self._dwconv(Ts,self.conv_k,self.cfg.qk_kernel)  # [#2]
        bdelta=torch.sigmoid(self.delta_beta) if self.cfg.use_delta else None           # [recall] delta write-gate (per-head) or None=additive
        if cap: Xl,S1,z1=lin_read(Ix,Ts,Cs,ch,beta=bdelta,ret_state=True)               # SINGLE brain read (hop1). lerp gate + hop2 removed (hop2 redundant: -hop2 ablation neutral §10[2]; gate's only consumer was hop2)
        else: Xl=lin_read(Ix,Ts,Cs,ch,beta=bdelta)
        if self.think_mode=='conv':
            if cap: Ta,tk=self._dwconv(Is,self.think_conv,self.cfg.think_kernel,cap=True)
            else: Ta=self._dwconv(Is,self.think_conv,self.cfg.think_kernel)  # [#8]
        else:
            if cap: Ta,S3,z3=lin_read(Is,Ts,Cs,ch,ret_state=True); tk=(S3,z3)
            else: Ta=lin_read(Is,Ts,Cs,ch)
        Ta=self.agg_t(Ta)                                                  # [#4] cross-cortex aggregation (self-think path)
        feat=torch.cat([Xl.permute(0,2,1,3).reshape(B,L,-1),Ta.permute(0,2,1,3).reshape(B,L,-1)],-1)   # [retrieval Xl ; think Ta] -> (B,L,2*H*dc)
        O=X+self.proj_o(feat); O=O+self.ffn(self.ln_ff(O))
        return (O,(cq,ck,S1,z1,tk)) if cap else O
    def step(self,x,state):                                                 # O(1) SINGLE-TOKEN incremental decode. x:(B,1,d), state from forward(cap=True). -> (O (B,1,d), new_state)
        cq,ck,S1,z1,tk=state; dk=self.dk; h=self.ln_in(x)
        Ix=self._h(self.to_Ix(h),dk); Ts=self._h(self.to_Ts(h),dk); Cs=self._h(self.to_Cs(h),self.dc)
        Is=self._h(self.to_Is(h),self.dc if self.think_mode=='conv' else dk)
        if self.cfg.qk_conv: Ix,cq=self._dwconv_step(Ix,self.conv_q,self.cfg.qk_kernel,cq); Ts,ck=self._dwconv_step(Ts,self.conv_k,self.cfg.qk_kernel,ck)
        bdelta=torch.sigmoid(self.delta_beta) if self.cfg.use_delta else None
        Xl,S1,z1=lin_read_step(Ix,Ts,Cs,S1,z1,beta=bdelta)
        if self.think_mode=='conv': Ta,tk=self._dwconv_step(Is,self.think_conv,self.cfg.think_kernel,tk)
        else: S3,z3=tk; Ta,S3,z3=lin_read_step(Is,Ts,Cs,S3,z3); tk=(S3,z3)
        Ta=self.agg_t(Ta); B=x.shape[0]
        feat=torch.cat([Xl.permute(0,2,1,3).reshape(B,1,-1),Ta.permute(0,2,1,3).reshape(B,1,-1)],-1)
        O=x+self.proj_o(feat); O=O+self.ffn(self.ln_ff(O)); return O,(cq,ck,S1,z1,tk)
class FeynmanHead(nn.Module):
    # [#3] PhotonLM Born head + Feynman path-integral + optional UNBOUNDED linear base.
    #   logits = base(h) + cap*tanh(log(|v|^2+eps)/cap)
    # WHY base: pure-Born logits live in [cap*tanh(log eps/cap), cap] -> max winner-vs-loser gap = cap - floor.
    # This forces a hard CE floor log(1+(V-1)*exp(-(cap-floor))): ~3.70 nats @ V=50257 (neo), ~0.18 @ V=256 (byte).
    # VERIFIED: pure-Born optimum CE caps at ~4.0 for neo; an unbounded linear head reaches ~0. So the base gives
    # unbounded sharpness (kills the floor) while the Born term keeps bounded rule-suppression. linear_base=False = pure-Born ablation.
    # head_eps: in-log floor bounds Born grad amplifier to 1/sqrt(eps). head_cap: bounds the Born correction (tanh).
    # born=False -> skip the path-integral path entirely (pure linear base): floor-free AND drops the born vr,vi (B,L,V)
    # activations (~2.4GB @neo) -> leanest head, best for large-vocab under VRAM pressure. born benefit is marginal (§10.5).
    def __init__(self,d,vocab,n_paths=64,head_eps=0.1,head_cap=5.0,linear_base=True,born=True):
        super().__init__(); self.eps=head_eps; self.cap=head_cap; self.linear_base=linear_base; self.born_on=born
        if born:                                                            # Born path-integral params (created only when born on)
            self.to_re=nn.Linear(d,n_paths,bias=False); self.to_im=nn.Linear(d,n_paths,bias=False)
            self.U_re=nn.Parameter(torch.eye(n_paths)+0.01*torch.randn(n_paths,n_paths)); self.U_im=nn.Parameter(0.01*torch.randn(n_paths,n_paths))
            self.out_re=nn.Linear(n_paths,vocab,bias=False); self.out_im=nn.Linear(n_paths,vocab,bias=False)
        self.base=nn.Linear(d,vocab,bias=False) if (linear_base or not born) else None   # unbounded sharpness (tie-able to embed); forced on when born off
    def forward(self,h):
        out=self.base(h) if self.base is not None else 0                    # unbounded linear logits (removes CE floor)
        if self.born_on:
            ar,ai=self.to_re(h),self.to_im(h)                               # (B,L,P) complex path amplitudes
            pr=ar@self.U_re.t()-ai@self.U_im.t(); pi=ar@self.U_im.t()+ai@self.U_re.t()   # propagate a'=U a (complex)
            vr=self.out_re(pr)-self.out_im(pi); vi=self.out_re(pi)+self.out_im(pr)       # project to vocab amps (complex)
            out=out+self.cap*torch.tanh(torch.log(vr*vr+vi*vi+self.eps)/self.cap)        # + bounded Born |amp|^2
        return out
def _clone_state(s): return None if s is None else (tuple(_clone_state(x) for x in s) if isinstance(s,tuple) else s.clone())   # deep-clone a layer state (None / nested (S3,z3) / tensors) into persistent buffers
def _copy_state(dst,src):                                                   # in-place copy src->dst preserving buffer addresses (CUDA-graph recurrence persists across replays)
    if dst is None: return
    if isinstance(dst,tuple):
        for d,sv in zip(dst,src): _copy_state(d,sv)
    else: dst.copy_(src)
class LerpLM(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg; self.embed=nn.Embedding(cfg.vocab_size,cfg.d_model)
        self.layers=nn.ModuleList([LerpLayer(cfg) for _ in range(cfg.n_layers)])
        self.ln_f=RMSNorm(cfg.d_model); self.head=FeynmanHead(cfg.d_model,cfg.vocab_size,cfg.n_paths,cfg.head_eps,cfg.head_cap,cfg.head_linear_base,cfg.head_born)
        self.apply(self._init)
        if cfg.tie and self.head.base is not None: self.head.base.weight=self.embed.weight   # weight tying: share linear base with embedding (0 extra params, standard quality win)
    def _init(self,m):
        if isinstance(m,nn.Linear):
            if getattr(m.weight,'_eye_init',False): pass
            else: nn.init.normal_(m.weight,0,0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0,0.02)
    def forward(self,idx,targets=None,grad_checkpoint=False):
        h=self.embed(idx)
        for layer in self.layers:
            h=_checkpoint(layer,h,use_reentrant=False) if (grad_checkpoint and self.training) else layer(h)
        logits=self.head(self.ln_f(h)); loss=None
        if targets is not None: loss=F.cross_entropy(logits.reshape(-1,self.cfg.vocab_size),targets.reshape(-1))
        return logits,loss
    @torch.no_grad()
    def generate(self,idx,max_new,temperature=1.0,top_k=None,fast='auto'):
        # fast: 'auto' (=graph on cuda else eager) | 'graph' (manual CUDA graph: records EXISTING eager kernels, NO Inductor codegen -> Windows-safe) |
        #       'compile' (torch.compile reduce-overhead: kernel-fusion + auto-cudagraph, Inductor path -> may crash ptxas on Windows, auto-falls back) | 'eager' (O(1)/tok python loop).
        # graph/compile kill per-token kernel-launch overhead (the short-seq wall-clock cost); fixed-size state makes the step STATIC-shape -> capturable (a structural edge over a growing KV-cache).
        self.eval(); cuda=idx.is_cuda
        if fast=='auto': fast='graph' if cuda else 'eager'
        if fast in('graph','compile') and cuda:
            try: return self._generate_fast(idx,max_new,temperature,top_k,compile_step=(fast=='compile'))
            except Exception as e: print(f"[generate] {fast} decode unavailable ({type(e).__name__}: {e}) -> eager fallback")
        h=self.embed(idx); states=[]                                        # EAGER INCREMENTAL: prefill ONCE (O(prompt)) capturing state, then O(1)/token. == full-forward logits (T9).
        for layer in self.layers: h,st=layer(h,cap=True); states.append(st)
        ll=self.head(self.ln_f(h))[:,-1]
        for _ in range(max_new):
            l=ll/max(temperature,1e-6)
            if top_k: v,_=torch.topk(l,min(top_k,l.size(-1))); l=l.masked_fill(l<v[:,[-1]],float('-inf'))
            tok=torch.multinomial(F.softmax(l,-1),1); idx=torch.cat([idx,tok],1)
            h=self.embed(tok)                                               # feed ONLY the new token
            for i,layer in enumerate(self.layers): h,states[i]=layer.step(h,states[i])
            ll=self.head(self.ln_f(h))[:,-1]
        return idx
    @torch.no_grad()
    def _generate_fast(self,idx,max_new,temperature,top_k,compile_step):
        # Static-buffer O(1)/token decode. step_once() reads persistent state buffers, advances them IN-PLACE, writes logits_buf -> fixed shapes/addresses every iter.
        # compile_step=False + cuda: manual CUDA graph (capture once, replay) -> no codegen, safe on Windows. compile_step=True: torch.compile(reduce-overhead) (Inductor; may crash -> caller falls back).
        # compile_step=False + cpu: runs step_once directly (NO capture) -> identical math to the graph, lets the buffer/in-place logic be unit-tested off-GPU.
        dev=idx.device; h=self.embed(idx); states=[]
        for layer in self.layers: h,st=layer(h,cap=True); states.append(st)
        logits=self.head(self.ln_f(h))[:,-1].clone()                        # prefill logits (predict 1st new token); cloned so logits_buf aliasing below is safe
        tok_buf=torch.zeros(idx.size(0),1,dtype=torch.long,device=dev); bufs=[_clone_state(st) for st in states]; logits_buf=torch.zeros_like(logits)
        def step_once():
            hh=self.embed(tok_buf)
            for i,layer in enumerate(self.layers): hh,ns=layer.step(hh,bufs[i]); _copy_state(bufs[i],ns)
            logits_buf.copy_(self.head(self.ln_f(hh))[:,-1])
        run=step_once; g=None
        if compile_step: run=torch.compile(step_once,mode='reduce-overhead')
        elif dev.type=='cuda':
            s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):                                      # warmup (required before capture)
                for _ in range(3): step_once()
            torch.cuda.current_stream().wait_stream(s)
            for i,st in enumerate(states): _copy_state(bufs[i],st)          # warmup advanced the state -> reset buffers to the prefill state
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): step_once()                          # capture (also advances state once)
            for i,st in enumerate(states): _copy_state(bufs[i],st)          # reset again -> decode starts from the true prefill state
        out=idx
        for _ in range(max_new):
            l=logits/max(temperature,1e-6)
            if top_k: v,_=torch.topk(l,min(top_k,l.size(-1))); l=l.masked_fill(l<v[:,[-1]],float('-inf'))
            tok=torch.multinomial(F.softmax(l,-1),1); out=torch.cat([out,tok],1); tok_buf.copy_(tok)
            g.replay() if g is not None else run()                         # advance state + refresh logits_buf
            logits=logits_buf
        return out
    def num_params(self): return sum(p.numel() for p in self.parameters())
if __name__=="__main__":
    torch.manual_seed(0)
    # tiny config for CPU verification (H=8 divisible by branch^levels=4). user full-scale: d_model=512,d_cell=64,n_lobe=4,n_cortex=64 (H=256).
    cfg=LerpConfig(vocab_size=65,d_model=48,d_cell=12,n_lobe=2,n_cortex=4,n_layers=3,chunk_len=16)
    m=LerpLM(cfg); H=cfg.n_lobe*cfg.n_cortex; dk=cfg.phi_expand*cfg.d_cell
    print(f"params={m.num_params()/1e3:.1f}K  H={H} dc={cfg.d_cell} dk={dk}(phi x{cfg.phi_expand}) layers={cfg.n_layers} think={cfg.think_mode} nest={cfg.nest_mode}")
    print("=== T0 chunked lin_read == cumsum reference (dk!=dv) ===")
    I,T,C=torch.randn(2,4,40,8),torch.randn(2,4,40,8),torch.randn(2,4,40,12)
    err=(lin_read(I,T,C,16)-_lin_read_ref(I,T,C)).abs().max().item(); print(f"  max|chunked-ref|={err:.2e} EXACT={err<1e-4}")
    x=torch.randint(0,65,(4,40)); y=torch.randint(0,65,(4,40))
    print("=== T1 forward + init loss (~ln65=4.174) ==="); lg,loss=m(x,y); print(f"  logits {tuple(lg.shape)} loss {loss.item():.4f}")
    print("=== T2 grads finite ==="); loss.backward(); print(f"  finite={all((p.grad is None) or torch.isfinite(p.grad).all() for p in m.parameters())}")
    print("=== T3 CAUSAL: perturb FUTURE token, earlier outputs unchanged (CRITICAL) ===")
    m.eval()
    with torch.no_grad():
        xb=torch.randint(0,65,(2,40)); o1,_=m(xb); p=27; xb2=xb.clone(); xb2[:,p]=(xb2[:,p]+5)%65; o2,_=m(xb2)
        bf=(o1[:,:p]-o2[:,:p]).abs().max().item(); af=(o1[:,p:]-o2[:,p:]).abs().max().item()
    print(f"  pos<{p}:{bf:.2e}(~0) pos>={p}:{af:.2e}(>0) CAUSAL={bf<1e-5 and af>1e-5}")
    print("=== T4 NOT EXPLODE: lin_read state size (no L dimension) ===")
    Bx,Lx=8,512; st=Bx*H*dk*cfg.d_cell*4/1e6; nai=Bx*H*Lx*dk*cfg.d_cell*4/1e6
    print(f"  B={Bx} L={Lx}: chunked state {st:.3f}MB  vs  naive cumsum ledger {nai:.1f}MB  ({nai/st:.0f}x saved)")
    print("=== T5 HierAgg(tt) at init == mean-pool ===")
    z=torch.randn(2,H,30,cfg.d_cell); a_tt=HierAgg(H,cfg.d_cell,2,2,'tt'); a_mn=HierAgg(H,cfg.d_cell,2,2,'mean')
    d_init=(a_tt(z)-a_mn(z)).abs().max().item(); print(f"  max|tt-mean|={d_init:.2e} (init equals mean: {d_init<1e-5})")
    print("=== T6 FeynmanHead grad amplifier bounded by head_eps (anti-cliff) ===")
    for ep in [1e-6,0.1]:
        hd=FeynmanHead(48,256,16,ep,5.0); hh=torch.randn(4,8,48,requires_grad=True)
        l=F.cross_entropy(hd(hh).reshape(-1,256),torch.randint(0,256,(32,))); l.backward()
        gn=hh.grad.norm().item(); print(f"  head_eps={ep:g}: input grad-norm={gn:.1f} (1/sqrt(eps)={1/math.sqrt(ep):.0f})")
    print("=== T7 phi-expand shapes (Q/K dim dk=2dc, value dc) ===")
    ly=m.layers[0]; print(f"  to_Ix out={ly.to_Ix.out_features}=H*dk={H*dk}  to_Cs out={ly.to_Cs.out_features}=H*dc={H*cfg.d_cell}  (Ix query dk, Cs value dc)")
    print("=== T8 LEARNS (byte shakespeare, cosine lr) ==="); m.train()
    import os; data=torch.tensor(list(open('./data/input.txt','rb').read())[:200000]) if os.path.exists('./data/input.txt') else torch.tensor(list((b"once upon a time there was a small cat. "*5000)))
    c2=LerpConfig(vocab_size=256,d_model=48,d_cell=12,n_lobe=2,n_cortex=4,n_layers=3,chunk_len=16); mm=LerpLM(c2)
    opt=torch.optim.AdamW(mm.parameters(),lr=2e-3,weight_decay=0.05); N=150
    for s_ in range(N):
        lr=2e-3*((min(1,(s_+1)/15)*0.5*(1+math.cos(math.pi*max(0,s_-15)/(N-15)))) if s_>=15 else (s_+1)/15)
        for gp in opt.param_groups: gp['lr']=lr
        ix=torch.randint(len(data)-65,(16,)); xb=torch.stack([data[i:i+64] for i in ix]); yb=torch.stack([data[i+1:i+65] for i in ix])
        _,l=mm(xb,yb); opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(mm.parameters(),1.0); opt.step()
        if s_%30==0 or s_==N-1: print(f"  step {s_:3d} lr {lr:.1e} loss {l.item():.4f} (init~ln256=5.545)")
    print("=== T9 INCREMENTAL DECODE == full forward (O(1)/token vs O(L^2)) ==="); m.eval()
    for kw in [{},{'think_mode':'lin'},{'use_delta':True}]:
        torch.manual_seed(0); cg=LerpConfig(vocab_size=65,d_model=48,d_cell=12,n_lobe=2,n_cortex=4,n_layers=3,chunk_len=16,**kw); mt=LerpLM(cg).eval()
        Bx,Lx=2,40; ii=torch.randint(0,65,(Bx,Lx))
        with torch.no_grad():
            fl,_=mt(ii); p=7; hh=mt.embed(ii[:,:p]); sts=[]
            for ly in mt.layers: hh,s_=ly(hh,cap=True); sts.append(s_)
            inc=[mt.head(mt.ln_f(hh))[:,-1]]
            for t in range(p,Lx):
                hh=mt.embed(ii[:,t:t+1])
                for i_,ly in enumerate(mt.layers): hh,sts[i_]=ly.step(hh,sts[i_])
                inc.append(mt.head(mt.ln_f(hh))[:,-1])
            e=(torch.stack(inc,1)-fl[:,p-1:]).abs().max().item()
        print(f"  {str(kw):34s} max|inc-full|={e:.2e} EXACT={e<1e-4}"); assert e<1e-4
    print("all model_lerp tests passed")
