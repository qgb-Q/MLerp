# MLerp
Brain-inspired, KV-free linear-attention LM — chunked causal linear attention in a four-level cell→cortex→lobe→whole hierarchy, with O(1)-state inference and no context-length limit.
# model_lerp

> A brain-inspired, KV-free linear-attention language model — chunked causal linear attention organized as a four-level `cell → cortex → lobe → whole` hierarchy, with constant-memory O(1)-state inference and no context-length limit.

`model_lerp` is a non-Transformer LM built on a single primitive: **`lin_read`**, a chunked causal *linear* attention that reads queries against a running `(dk, dv)` matrix ledger. Because the ledger carries state across chunks instead of materializing per-token keys/values, the sequence dimension is removed entirely — state is `O(B·H·dk·dv)`, **not** `O(B·H·L·dk·dv)`. The result: it **never numerically explodes**, decodes at **O(1) per token**, and has **no positional ceiling** (no position embedding; blocks are length-independent).

## Highlights

- **KV-free linear attention** — `lin_read` is mathematically identical to the cumulative-sum reference (verified to fp32, 7.2e-7) but carries a fixed-size ledger, not a growing KV cache.
- **Never explodes** — additive chunked state with no `L` dimension; bounded memory regardless of sequence length.
- **Inference is the payoff** — O(L) compute, **O(1) recurrent state**, constant VRAM, and **no context limit** (decode arbitrarily long via incremental `step`).
- **Brain-inspired hierarchy** — heads are grouped as cortices → lobes; cross-cortex mixing is a learnable **HierAgg** (TT-RNN core) replacing mean-pool.
- **FeynmanHead** — Born-rule-capable output head with an unbounded linear base (on by default; the linear base removes the large-vocab cross-entropy floor that pure-Born imposes).
- **CUDA-graph / `torch.compile` fast decode**, with an eager fallback.

## Architecture (per layer)

Four projections feed one scan:
- `to_Ix` — token query
- `to_Ts / to_Cs / to_Is` — whole-brain key / value / self-think

`Xl = lin_read(Ix ; Ts, Cs)` is the single causal scan (token reads brain memory). Short causal conv on Q/K (selectivity), conv- or linear-mode self-think, then RMSNorm + FFN → FeynmanHead. Optional DeltaNet erase-before-write (`--use_delta`).

**Production config** (`train.py` defaults): `d_model=1024, d_cell=64, n_lobe=4, n_cortex=4 (H=16), n_layers=12, chunk_len=256`, neo BPE (GPT-2, 50257 vocab) → **253.22M params**, linear head.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # 2.6 + CUDA 12.4
pip install numpy matplotlib datasets tiktoken
```

## Quickstart

```bash
# 1. Build the corpus (streams FineWeb sample-10BT -> out_lerp/{train,val}_neo.bin; caps raw text, never pulls the full TBs)
python prepare_data.py                    # --max_gb 5 to pull less; --tokenizer byte for byte-level

# 2. Train (253M; reads out_lerp/*.bin; bs2 x block2048 x accum4 = 16384 eff tok/step; torch.compile on)
python train.py                           # --resume to continue; --no_compile to disable; --chunk_len to sweep

# 3. Generate (O(1)/token incremental decode -- no context window)
python generate.py --prompt "Once upon a time" --max_new 300 --top_p 0.95
python generate.py out_lerp/ckpt.pt --ppl out_lerp/val_neo.bin   # held-out perplexity (the objective metric)
```

Checkpoints land in `out_lerp/` (`ckpt.pt`, `ckpt_best.pt`); training auto-streams more FineWeb when the corpus is exhausted (`--no_topup` to disable).

## Controlled comparison vs a matched Transformer

`anchoring_group.py` trains a standard pre-LN Transformer **param-matched** to model_lerp on the **same** `.bin` and budget, then benchmarks both side by side:

```bash
python anchoring_group.py                              # train the matched anchor -> out_anchor/
python anchoring_group.py --compare out_lerp/ckpt.pt   # 8-panel comparison.png (params/loss/ppl/throughput/VRAM/inference scan)
python emergent_probe.py --lerp out_lerp/ckpt.pt --anchor out_anchor/ckpt.pt --bin_dir out_lerp   # 6 behavioral probes
```

## Files

| File | What |
|---|---|
| `model_lerp.py` | Architecture: `LerpLM` / `LerpConfig`, `lin_read`, HierAgg, FeynmanHead, O(1) `step` + CUDA-graph `generate`. |
| `train.py` | Trainer: bf16 AdamW, cosine+warmup, grad-accum/ckpt, eval/resume, dynamic FineWeb top-up, loss curves. |
| `prepare_data.py` | Streams FineWeb → `out_lerp/{train,val}_<tok>.bin` (capped, never the full set). |
| `generate.py` | Sampler + honest evaluator (incremental decode, held-out ppl, cross-distribution battery). |
| `anchoring_group.py` | Param-matched Transformer baseline + side-by-side comparison. |
| `emergent_probe.py` | Six architecture-grounded behavioral probes (length-extrapolation, induction, state trajectory, …). |
| `test.py` | 72-test diagnostic suite. |
| `ablation.py` | phi / DeltaNet ablations. |
| `model_lerp_research_record.md` | Full design + empirical record. |

## Status (honest)

At 253M params on FineWeb vs a param-matched Transformer: validation loss is **slightly behind** (≈4.14 vs 3.98 nats, +17% ppl) and training is **slower** (≈12k vs 18k tok/s — the chunk-scan is sequential, intrinsically lower-MFU). The **validated payoff is inference scaling**: throughput stays flat out to 16k tokens at constant ~3.7 GB VRAM with O(L) latency, while the Transformer is hard-capped at its 2048 context and its KV cache grows linearly.

Caveat: these inference properties are **shared with the linear-attention family** (Mamba / RWKV / GLA / RetNet); behavioral probes did **not** isolate a distinct contribution from the brain-inspired hierarchy itself. See `model_lerp_research_record.md` §10.14–§10.16 for the full, unembellished analysis.

## License

TBD.
