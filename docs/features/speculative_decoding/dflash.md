# DFlash (Block-Diffusion Draft Models)

DFlash is a speculative decoding method where a small **block-diffusion draft model**
proposes a block of tokens in a single non-autoregressive forward pass, conditioned on
hidden states from intermediate layers of the target model. The target model then
verifies the block as usual — decoding remains **lossless** (greedy outputs match the
target model token-for-token).

This fork adds DFlash support for **NemotronH targets (Nemotron Ultra 550B)**, on top of
the existing `method: "dflash"` proposer plumbing. Key pieces:

- `vllm/model_executor/models/nemotron_h.py` — auxiliary hidden-state taps
  (`DFLASH_AUX_HIDDEN_STATE_LAYERS`) and offline extraction hooks (`DFLASH_EXTRACT_PATH`)
- `vllm/model_executor/models/qwen2.py` — `relu2` activation (relu²(gate)·up) used by the
  DFlash draft MLP
- `vllm/v1/spec_decode/` — dflash mask-token plumbing; the draft's non-causal attention
  requires a **bf16 KV cache** (see [Hard rules](#hard-rules))

Reference results below come from a production campaign on Nemotron Ultra 550B (NVFP4,
TP8, 1× B200 node) where DFlash beat native MTP (k=5) in **all four** production
concurrency regimes.

## Step 1 — Get a working vLLM

There are two paths. The **verified path** is the exact stack behind every certified
number below; the **fresh-pull path** builds this fork from source.

### Path A — verified reproduction stack (recommended, exact benchmark parity)

The certified numbers were produced with a **vLLM v0.22 container + 2 files from this
fork bind-mounted over it** (the 0.22 image has the `dflash` method natively; the two
overlays add the relu2 draft activation and bf16-KV enforcement). On the Nous cluster
everything already exists — no build step at all:

| Asset | Cluster path |
| --- | --- |
| Container (SIF) | `/home/dakota/sif/vllm-v0.22.0.sif` |
| Overlay patches | `/home/phuc/workspace/rl/small_prs/pr015_dflash_nemotron_ultra/patches_v022/` |
| **Ready-to-run sbatch, n≤8** (champ-b16, k13) | `.../pr015_dflash_nemotron_ultra/scripts/PROD_serve_dflash_lowconc.sh` |
| **Ready-to-run sbatch, n≥32** (champ-b8, k5; edit k→3 for n=128) | `.../pr015_dflash_nemotron_ultra/scripts/PROD_serve_dflash_highconc.sh` |
| Benchmark script (produced all numbers below) | `.../pr015_dflash_nemotron_ultra/scripts/bench_http_compare.py` |
| Raw bench results + campaign worklog | `.../pr015_dflash_nemotron_ultra/output/bench_http_*.json`, `/home/phuc/workspace/moe/worklogs/2026-07-03/dflash-vs-mtp-nemotron-ultra-550b-benchmark-campaign.md` |

To reproduce the best runs: `sbatch PROD_serve_dflash_lowconc.sh` (or `_highconc`),
wait for the endpoint file it writes, then run `bench_http_compare.py --max-tokens 512`
against it directly (never through a router). That is the entire recipe behind the table
in [Tips](#tips--picking-the-draft-and-k).

The bind-mount lines (already inside the PROD scripts), if you assemble your own launch:

```text
--bind patches_v022/llm_base_proposer.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/llm_base_proposer.py
--bind patches_v022/qwen2.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen2.py
```

### Path B — fresh pull of this fork

```bash
git clone https://github.com/NousResearch/vllm.git
cd vllm
# main = upstream vllm + DFlash-for-NemotronH commits on top
VLLM_USE_PRECOMPILED=1 uv pip install -e .   # or your usual build path
```

Runtime-validated on B200 (greedy outputs verified sensible; see caveats). To match the
benchmark environment, three environment details matter — all discovered the hard way:

1. **Driver / torch mismatch:** the precompiled wheel pulls torch cu130; on driver-570
   (CUDA 12.8) nodes CUDA init fails with "driver too old". Fix: put CUDA 13
   forward-compat libs (libcuda 580, extractable from any recent NGC/vLLM container's
   `/usr/local/cuda/compat`) on `LD_LIBRARY_PATH`.
2. **Pin FlashInfer to the benchmark triplet:** `flashinfer-python==0.6.11.post2
   flashinfer-cubin==0.6.11.post2 flashinfer-jit-cache==0.6.11.post2+cu130`.
   The default 0.6.13 resolution (without the jit-cache wheel) segfaulted in fp8_gemm
   autotuning on this driver combo; the pinned triplet runs the full autotune cleanly,
   identical to the benchmarked container.
3. **Keep `--compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}'`**
   (it is part of the production recipe anyway): vLLM main's fused-allreduce pass calls
   a newer flashinfer `allreduce_fusion` signature than 0.6.11 provides
   (`weight_bias` kwarg) and crashes at CUDA-graph capture with it enabled.

## Step 2 — Get the draft weights

Two production draft checkpoints exist for Nemotron Ultra 550B (internal cluster paths):

| Draft | Path | Use for |
| --- | --- | --- |
| **b16** | `/home/phuc/workspace/rl/small_prs/pr015_dflash_nemotron_ultra/output/champ_b16_100k_paperopt/checkpoints/checkpoint_best/` | low concurrency (n≤8) |
| **b8** | `/home/phuc/workspace/rl/small_prs/pr015_dflash_nemotron_ultra/output/champ_b8_100k_paperopt/checkpoints/checkpoint_best/` | high concurrency (n≥32) |

Copy `config.json`, `config.py`, `model-*.safetensors`, `model.safetensors.index.json`
(~5.6 GB per draft). Skip `optimizer_state_dict.pt` / `scheduler_state_dict.pt` (training
state, ~6 GB you don't need for inference).

Target models (cluster paths): NVFP4 (all certified numbers)
`/home/shared/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4/`; a BF16 target also
exists under `.../pr007*/ckpt_real_052726/` (slower at n≤8 — 4× weight bandwidth — use
NVFP4 to reproduce the table). The working local checkout of this fork lives at
`/home/phuc/workspace/nous-vllm-fork/` (branch
`phuc/dflash-nemotron-ultra-v0.20.1-20260629` is the battle-tested v0.20.1 line used for
early BF16 benches, incl. `DFLASH_MIXED_ATTN=1`; `main` is the current rebased line).

### b16 vs b8 — what's actually different?

Both are ~4B-param, 5-layer, D=8192 drafts (qwen3-style layers, `hidden_act=relu2`,
GQA 64/2 heads) trained on the same 100K on-policy conversations with the same optimizer
recipe. The **only architectural difference is the diffusion block size** — how many
tokens the draft denoises (and can therefore propose) per verify step:

| | b16 | b8 |
| --- | --- | --- |
| `block_size` (max useful k) | **16** (k ≤ 15) | **8** (k ≤ 7) |
| Training γ (noise schedule) | 7 | 4 |
| Training anchors/seq | 4096 | 8192 |
| Val position-1 acceptance | 89.9% | **91.1%** |
| Val position-7 acceptance | 61.8% | 61.1% |
| Val full-block acceptance | 62.7% (of 16) | **75.2%** (of 8) |
| Deepest useful position | pos-15: 46.3% | pos-7: 61.1% |

Trade-off in one sentence: **b16 speculates deeper, b8 speculates slightly more
accurately per position.** At low concurrency the verify step is nearly free (GPUs are
idle), so depth wins → b16 with large k. At high concurrency every rejected draft token
displaces a real token in the batch, so shallow-and-accurate wins → b8 with small k.
A block-8 draft has a hard cliff at position 8 (acceptance ≈3% beyond its block), so
never set k > block_size−1 of the draft you deploy.

## Step 3 — Serve

Take your existing target-model serve command and change **two things**: swap the
speculative config to dflash, and force the KV cache to bf16.

Full production example (Nemotron Ultra 550B NVFP4, TP8, 1×B200 node):

```bash
vllm serve /path/to/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 \
  --trust-remote-code --tensor-parallel-size 8 --enable-expert-parallel \
  --max-num-seqs 128 --max-model-len 1048576 --max-num-batched-tokens 32768 \
  --enable-prefix-caching --enable-chunked-prefill \
  --kv-cache-dtype bfloat16 \
  --mamba-ssm-cache-dtype float16 --mamba-backend flashinfer \
  --enable-mamba-cache-stochastic-rounding --mamba-cache-philox-rounds 5 \
  --compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}' \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --reasoning-parser-plugin /path/to/target/ultra_v3_reasoning_parser.py \
  --reasoning-parser ultra_v3 \
  --speculative-config '{"model": "/path/to/draft/checkpoint_best", "method": "dflash", "num_speculative_tokens": 13}' \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}' \
  --host 0.0.0.0 --port 8000 --served-model-name dflash-nvfp4-ultra
```

Useful env (matches the reference numbers): `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`,
`VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm`, `VLLM_USE_FLASHINFER_MOE_FP4=1`,
`VLLM_FLASHINFER_MOE_BACKEND=latency`, `SAFETENSORS_FAST_GPU=1`,
`VLLM_WORKER_MULTIPROC_METHOD=spawn`.

Load time is ~10–15 min (lots of CUDA-graph capture at 8 GPUs).

## Step 4 — Verify before trusting anything

1. **Startup log** shows `method=dflash`.
2. **Acceptance telemetry**: after a few requests, mean accepted tokens per verify step
   ("τ") should be ≈7 for b16@k13 and ≈2.5–3 for b8@k3.
   **If τ < 1.5, the `relu2` activation isn't active** (the draft loads fine but produces
   garbage) — check Step 1 / the qwen2 overlay.
3. **Losslessness spot-check** (once per new serve config): send ~10 diverse prompts with
   `temperature=0` to this endpoint and to a non-speculative serve of the same target.
   Outputs must match **token-for-token**. Any mismatch is a setup bug, not sampling noise.

## Tips — picking the draft and k

`num_speculative_tokens` (k) per concurrency is the whole tuning game:

| Workload | Draft | k | Measured agg tok/s (Ultra 550B, 512-tok, B200×8) |
| --- | --- | --- | --- |
| n=1 (latency / vibe test) | b16 | 11 | ~387 |
| n≤8 (interactive) | b16 | 13 | ~1,675 |
| n≈32 (RL rollouts) | b8 | 5 | ~3,602 |
| n≥128 (synthetic data gen) | b8 | 3 | ~4,748 |

Intuition: idle GPUs → speculate deep (large k, deep draft); saturated batch → every
rejected token displaces a real one, so shrink k and use the more-accurate shallow draft.
One endpoint cannot be optimal across both regimes — run two endpoints (b16@k13 and
b8@k5/k3) if you serve both.

### Hard rules

- `--kv-cache-dtype bfloat16` is **mandatory**. The DFlash draft uses non-causal block
  attention, which is incompatible with fp8 KV cache. This is the one real cost vs an
  fp8-KV MTP config (less KV headroom at very long context).
- Never set k ≥ the draft's `block_size` (b8: k≤7; b16: k≤15, throughput peak at 13).
- Keep `dflash_config.mask_token_id` in the draft's `config.json` — the proposer needs it.

### Benchmarking (how the table above was measured)

- Tool: `bench_http_compare.py` from the pr015 scripts dir; `--max-tokens 512`,
  concurrency levels matching the regime (e.g. `--levels 1 8 8 8 8 8` for repeated n=8
  cells). All numbers are **clean-node medians over ≥3 in-serve reps**.
- Benchmark direct-to-node, never through a router/load-balancer.
- Warm up first; short cells (n=8 ≈ 3 s) vary ±15–30 % run-to-run → report medians over ≥3 reps.
- If tok/s drops while acceptance telemetry (τ) is unchanged, suspect a degraded node, not the model.
- If the target emits `reasoning_content` (e.g. Nemotron Ultra), your token counting must
  include it or throughput will look artificially low (`bench_http_compare.py` handles this).

## Training your own DFlash draft

Draft training lives in [NousResearch/speculators](https://github.com/NousResearch/speculators)
(includes Nemotron Ultra 550B target support: NemotronH weight mapping and 5-target-layer
hidden-state extraction). The short version of what mattered, in order of impact:
**on-policy training data** (generations from the target model itself) ≫ data scale ≫
block-16 + γ=7 ≫ paper optimizer (lr 6e-4, cosine, 4 % warmup, 6 epochs). Hidden states
for training can be extracted with this fork's `DFLASH_EXTRACT_PATH` /
`DFLASH_AUX_HIDDEN_STATE_LAYERS` hooks in `nemotron_h.py`.
