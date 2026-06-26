# DigenRL TSP in UniRL — integration + measured speedup

**Goal:** add a DigenRL technique to UniRL and measure the speedup, on `unirl_video_zw1`.
**Technique chosen:** Time-Step Parallelism (TSP) — the most tractable, math-equivalent DigenRL idea.
**Worktree:** `.workdir/unirl_digenrl` (branch `feat/digenrl-tsp`, off main `1e3db3e`).
**Hardware:** zw1 node `29.162.233.209`, H20-96GB, WAN2.1-T2V-1.3B (cached).

## What TSP is (and where it lands in UniRL)
UniRL's diffusion *replay* (the train forward, `models/<m>/diffusion.py::DiffusionStage.replay`)
recomputes the policy log-prob by running the transformer forward **once per selected denoise
step** — a literal `for step_idx in target:` loop. Each step is an independent forward over the
stored (fixed) trajectory, so the K selected steps can be **batched into ONE forward of K·B**.
That is TSP. The SDE log-prob math (`strategy.denoise`) stays per-step (cheap); only the expensive
transformer forward is batched. Math-equivalent (verified: max|Δ| ~ bf16 noise, ≤5e-2).

## Measured speedup (replay forward+backward, WAN2.1-1.3B, latent [1,16,5,30,52])

### Single GPU (no sharding) — `bench_tsp.py`
| K | seq | tsp | speedup |
|---|-----|-----|---------|
| 1 | 410ms | 412ms | 1.00x |
| 2 | 822ms | 820ms | 1.00x |
| 4 | 1645ms | 1606ms | 1.02x |
| 8 | 3286ms | 3221ms | 1.02x |

Also tested at a large latent ([1,16,21,60,104], ~32k tokens): **1.00x**.

→ **On a single GPU, TSP gives ~0% gain: the diffusion forward is already compute-bound, even at
batch 1 / small latents.** Batching the batch-dim does the same FLOPs, so no win.

### FSDP-sharded (5 GPUs, FULL_SHARD) — `bench_tsp_fsdp.py`
| K | seq | tsp | speedup | peak mem |
|---|-----|-----|---------|----------|
| 1 | 391ms | 391ms | 1.00x | 13.5GB |
| 2 | 770ms | 737ms | **1.04x** | 26.2GB |
| 4 | 1519ms | 1404ms | **1.08x** | 51.6GB |
| 8 | — | — | OOM | ~100GB |

→ **Under FSDP, TSP gives a small, K-growing gain (1.04x→1.08x)** — this is the all-gather
amortization DigenRL describes: K sequential forwards = K param all-gathers; one batched forward =
1 all-gather. But for a 1.3B model the all-gather is cheap vs compute, so the gain is modest, and
**peak memory grows ~K×** (batched holds K× activations for the backward), capping usable K (K=8 OOMs).

## Bottom line
- **TSP alone, on accessible single-node hardware with a 1.3B video model: ~1.0–1.08x.** The benefit
  is real but marginal and bounded by the K× memory cost; it only matters once the per-step
  all-gather is a large fraction of step time — i.e. **much larger sharded models (13–20B)**.
- DigenRL's headline **1.56–2.1×** is NOT from TSP's forward-batching alone. It is the *system-level*
  combination — GAP (generation-axis micro-batching) + TAG (trainer-assisted generation) + TCSS
  (one-step-stale async) — filling pipeline bubbles in a **disaggregated** generator/trainer split.
  Those require re-architecting UniRL's (colocated) rollout/train loop, not a drop-in forward change.
- Recommended next step if pursuing this: implement TSP in replay for the **large** video models
  (Hunyuan-Video-13B) under FSDP where all-gather dominates, and/or prototype the GAP+TCSS
  disaggregated pipeline — that is where the 1.5–2× lives.

Scripts: `.clusters/.script/bench_tsp.py` (single-GPU), `.clusters/.script/bench_tsp_fsdp.py` (FSDP).
