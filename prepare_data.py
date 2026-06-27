"""Stream FineWeb (general high-quality web text) and tokenize to the .bin memmap that train.py consumes
(out_dir/{train,val}_<tok>.bin). Replaces the TinyStories toy set. Reuses train.py's EXACT tokenizers (import).
  python prepare_data.py                                    # FineWeb sample-10BT, neo BPE, cap ~10GB of raw text -> ./out_lerp
  python prepare_data.py --max_gb 5                         # pull less
  python prepare_data.py --tokenizer byte                   # byte-level (uint8)
Needs network + `pip install datasets`. STREAMING: never downloads the multi-TB full set; it reads shards on the fly and
STOPS after --max_gb of raw text (FineWeb is English -> ~1 byte/char), so the pull is bounded. Streamed bytes are compressed
parquet (~3-4x smaller than text); the output .bin is uint16 (2 bytes/token, ~half the text size).
~10GB text ~= 2.4B neo tokens ~= 4.8GB bin (plenty for single-GPU training). val = first --val_tokens of the stream (disjoint).
"""
import os,argparse,time,json
import numpy as np
from train import get_tok                                   # reuse EXACT byte/neo tokenizers so .bin == what train.py expects
REPOS={'fineweb':'HuggingFaceFW/fineweb','fineweb-edu':'HuggingFaceFW/fineweb-edu'}   # both expose a 'text' col + sample-10BT config
def stream_to_bin(examples,tok,dt,ftr,fva,max_chars,val_tokens,text_col,buf_flush=2_000_000,log_every=50000):
    buf=[]; nval=0; ntr=0; doc=0; nchars=0; t0=time.time()
    def flush(arr):                                          # route a token block: fill val first (disjoint prefix), rest to train
        nonlocal nval,ntr
        if nval<val_tokens:
            k=min(len(arr),val_tokens-nval)
            if k: arr[:k].tofile(fva); nval+=k
            if len(arr)>k: arr[k:].tofile(ftr); ntr+=len(arr)-k
        else: arr.tofile(ftr); ntr+=len(arr)
    for ex in examples:
        txt=ex[text_col]; nchars+=len(txt)                  # raw-text size proxy (FineWeb=English ~1B/char) -> bounds the pull
        ids=tok.encode(txt); ids.append(tok.eot); buf+=ids; doc+=1
        if len(buf)>=buf_flush:
            flush(np.asarray(buf,dtype=dt)); buf=[]
            if doc%log_every==0: print(f"  {doc} docs | {nchars/1e9:.2f}/{max_chars/1e9:.1f} GB text | val {nval/1e6:.1f}M train {ntr/1e6:.1f}M tok | {time.time()-t0:.0f}s")
        if nchars>=max_chars: break
    if buf: flush(np.asarray(buf,dtype=dt))                 # tail
    return ntr,nval,doc,nchars
def _meta_path(out_dir,tok_name): return os.path.join(out_dir,f'data_meta_{tok_name}.json')
def topup_bin(out_dir,tok_name,add_gb=10.0):
    # DYNAMIC top-up: resume the stream past already-pulled docs (skip -> fresh data, NO repeats) and APPEND ~add_gb more
    # tokens to train_<tok>.bin. Returns tokens appended, or -1 if no meta sidecar (run prepare_data.py first to create it).
    mp=_meta_path(out_dir,tok_name)
    if not os.path.exists(mp): return -1
    meta=json.load(open(mp)); tok=get_tok(tok_name); dt=np.uint8 if tok.vocab_size<=256 else np.uint16
    tr=os.path.join(out_dir,f'train_{tok_name}.bin')
    from datasets import load_dataset
    ds=load_dataset(meta['repo'],name=meta['config'],split='train',streaming=True).skip(meta['docs_pulled'])   # skip consumed docs -> fresh shards (cost grows per top-up; raise --topup_gb for fewer, larger pulls)
    ftr=open(tr,'ab'); dn=open(os.devnull,'wb')                                                                 # APPEND to train; val_tokens=0 routes everything to train (val stays fixed)
    try: ntr,_,doc,_=stream_to_bin(ds,tok,dt,ftr,dn,int(add_gb*1e9),0,meta['text_col'])
    finally: ftr.close(); dn.close()
    meta['docs_pulled']+=doc; meta['train_tokens']=meta.get('train_tokens',0)+ntr; json.dump(meta,open(mp,'w'),indent=2)
    return ntr
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='fineweb',help="key in REPOS (fineweb|fineweb-edu) or a raw HF repo id")
    p.add_argument('--config',default='sample-10BT',help="HF config (sample-10BT|sample-100BT|default|CC-MAIN-...)")
    p.add_argument('--tokenizer',default='neo',choices=['byte','neo']); p.add_argument('--out_dir',default='./out_lerp')
    p.add_argument('--text_col',default='text'); p.add_argument('--max_gb',type=float,default=10.0,help="STOP after this many GB of raw text -> bounds the download (never pulls the full TBs)"); p.add_argument('--val_tokens',type=int,default=5_000_000)
    a=p.parse_args(); os.makedirs(a.out_dir,exist_ok=True)
    repo=REPOS.get(a.dataset,a.dataset); tok=get_tok(a.tokenizer); dt=np.uint8 if tok.vocab_size<=256 else np.uint16
    tr=os.path.join(a.out_dir,f'train_{tok.name}.bin'); va=os.path.join(a.out_dir,f'val_{tok.name}.bin')
    if os.path.exists(tr): print(f"[prep] {tr} already exists ({os.path.getsize(tr)/1e6:.0f}MB) -> delete it to re-prep. Aborting."); return
    from datasets import load_dataset
    print(f"[prep] streaming {repo} ({a.config}) | tok={tok.name} vocab={tok.vocab_size} dtype={np.dtype(dt).name} | cap={a.max_gb:.1f}GB text, val={a.val_tokens/1e6:.0f}M -> {a.out_dir}")
    ds=load_dataset(repo,name=a.config,split='train',streaming=True)
    ftr=open(tr,'wb'); fva=open(va,'wb'); t0=time.time()
    try: ntr,nval,doc,nchars=stream_to_bin(ds,tok,dt,ftr,fva,int(a.max_gb*1e9),a.val_tokens,a.text_col)
    finally: ftr.close(); fva.close()
    print(f"[done] {nchars/1e9:.2f}GB text -> train {ntr/1e6:.2f}M tok ({os.path.getsize(tr)/1e6:.0f}MB) + val {nval/1e6:.2f}M tok | {doc} docs | {time.time()-t0:.0f}s")
    json.dump({'repo':repo,'config':a.config,'text_col':a.text_col,'tokenizer':tok.name,'dtype':np.dtype(dt).name,'docs_pulled':doc,'val_tokens':a.val_tokens,'train_tokens':ntr},open(_meta_path(a.out_dir,tok.name),'w'),indent=2)   # meta -> train.py auto-topup can resume the stream
    print(f"[next] python train.py --tokenizer {tok.name}   # finds these .bin in {a.out_dir} and skips re-tokenizing; auto-tops-up +10GB when consumed (--no_topup to disable)")
if __name__=='__main__': main()
