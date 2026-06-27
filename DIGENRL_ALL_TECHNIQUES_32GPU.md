# DigenRL all 4 techniques in UniRL — 32-GPU study

Goal: add DigenRL's 4 techniques (TSP/GAP/TAG/TCSS) to UniRL and measure each one's
speedup, on `unirl_video_zw1`, 32-GPU.

Worktree `.workdir/unirl_digenrl` (branch `feat/digenrl-tsp`). Hardware: zw1, 4 nodes ×
8× H20-96GB = **32 GPUs**. Model: WAN2.1-T2V-1.3B (cached), FSDP FULL_SHARD.

## TL;DR — what each technique is, and what we could actually measure
| Technique | What it does | Status in UniRL | Speedup |
|---|---|---|---|
| **TSP** | batch the K selected denoise steps into 1 replay forward | **implemented** (SD3 replay, `UNIRL_TSP_REPLAY`) | **MEASURED 1.07–1.11×** |
| **GAP** | micro-batch along the generation axis → more pipeline units | **implemented** (`GAPPlanner`, train-side) | needs disaggregated runtime; ceiling below |
| **TAG** | idle trainer GPUs help run generation | integration point identified (separate-mode work-steal) | needs disaggregated runtime |
| **TCSS** | 1-step-stale async, per-trajectory single snapshot | already ~present (`AsyncARTrainer` staleness buffer, AR only) | needs async **diffusion** runtime |

**Only TSP is a drop-in we could measure directly.** GAP/TAG/TCSS are system-level
pipeline techniques: their speedup only exists once rollout and training run as a
**disaggregated, overlapped** pipeline — which UniRL has for AR (`AsyncARTrainer`) but
not for diffusion. So for those we measured the *primitives that bound their benefit*
and give a theoretical ceiling, not an end-to-end number.

## Measured primitives (32-GPU, consistent across all 4 nodes)
Per-node 8-GPU FSDP, WAN2.1-1.3B, latent [1,16,5,30,52], 4 nodes run identically:
- **generator step** (one denoise forward, no-grad, CFG): **130 ms**
- **trainer step** (one selected-step replay fwd+bwd): **~380 ms**
- **TSP** (replay forward-batching, fwd+bwd): K=2 → **1.07–1.11×**, K=4 → **1.07×**

All 4 nodes (32 GPUs) reported gen_step 129.8–130.5 ms and TSP 1.067–1.115× — i.e. the
per-replica speedup is stable at 32-GPU scale (RL scales as DP replicas).

## TSP — the real result
- **1.07×** on 8/32-GPU FSDP for WAN2.1-1.3B. Single-GPU (no sharding) was ~1.00×
  (forward is compute-bound). The small FSDP gain is all-gather amortization (K forwards
  → 1). It grows with K but peak memory grows ~K× (K=8 OOMs). For a **1.3B** model the
  all-gather is cheap vs compute, so the gain is modest; it scales with model size (the
  13–20B models in the DigenRL paper see more).
- Code: `unirl/models/sd3/diffusion.py::SD3DiffusionStage.replay`, env-gated, math-equivalent.

## GAP / TAG / TCSS — theoretical ceiling from measured primitives
Computed by `digenrl_pipeline_model.py` from gen=130ms/step, train=380ms/step. These are
**ceilings, not measurements** — and they only beat colocated once TAG-style work-stealing
removes the disaggregation split penalty (naive fixed-split disaggregation ≈ colocated).

For a gen-bound config (T=50 denoise, K=16 trained steps): generator 6.5s vs trainer 5.7s
per RL step.
- colocated baseline: 12.2s (1.00×)
- + GAP (micro-pipeline, M=8): 8.0s → **1.52×** (ceiling)
- + TCSS (async, removes tail bubble): 6.5s → **1.87×** (= overlap ceiling)
- + TAG (work-steal upper bound): → **up to 2.0×**

The overlap ceiling = (gen+train)/max(gen,train); it ranges ~1.4–1.9× depending on the
train-step fraction K. **Realizing any of it requires the disaggregated diffusion pipeline
that UniRL does not yet have** — building it (async diffusion trainer + rollout buffer +
work-stealing) is the actual project for GAP/TAG/TCSS; this study delivers TSP + the
primitives + the GAP planner + the integration map.

## What was implemented (code, this branch)
- `unirl/models/sd3/diffusion.py` — TSP replay fast-path (prior commit).
- `unirl/train/stack/planner/gap.py` + `__init__.py` — `GAPPlanner` (generation-axis,
  single-sample micros; the train-side half of GAP).
- Benchmarks: `.clusters/.script/bench_digenrl.py` (primitives + TSP, FSDP),
  `bench_tsp.py` / `bench_tsp_fsdp.py` (TSP), `digenrl_pipeline_model.py` (ceiling model),
  `nccl_smoke.py` + `launch_multinode.sh` (multinode harness).

## Honest caveats
- Raw multinode (single 32-GPU NCCL group) **did not come up** — cross-node NCCL-over-socket
  hangs at ring build (cluster uses RDMA/ray, not raw TCP NCCL). 32-GPU here = 4×8 DP
  replicas run concurrently (valid for per-replica speedup, which is what these techniques
  change). A true 32-GPU single-group run would need the ray/RDMA path UniRL uses.
- GAP/TAG/TCSS numbers are model ceilings from measured primitives, NOT end-to-end runs.
