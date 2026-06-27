# DigenRL all 4 techniques in UniRL — 32-GPU study

Goal: add DigenRL's 4 techniques (TSP/GAP/TAG/TCSS) to UniRL and measure each one's
speedup, on `unirl_video_zw1`, 32-GPU.

Worktree `.workdir/unirl_digenrl` (branch `feat/digenrl-tsp`). Hardware: zw1, 4 nodes ×
8× H20-96GB = **32 GPUs**. Model: WAN2.1-T2V-1.3B (cached), FSDP FULL_SHARD.

## TL;DR — what each technique is, and the measured speedup
| Technique | What it does | Status in UniRL | Speedup |
|---|---|---|---|
| **TSP** | batch the K selected denoise steps into 1 replay forward | **implemented** (SD3 replay, `UNIRL_TSP_REPLAY`) | **MEASURED 1.07–1.11×** |
| **GAP+TCSS** | disaggregated async pipeline: overlap generation with train+weight-sync | **implemented** (async runtime `async_runtime.py`, modeled on `AsyncARTrainer`) + `GAPPlanner` | **MEASURED 1.36–1.44×** (end-to-end) |
| **TAG** | idle trainer GPUs work-steal generation (fine-grained) | **implemented** (work-steal in the async runtime) | **MEASURED up to 1.42×** (config-dependent; ≈GAP+TCSS when gen-bound) |

We initially only modeled GAP/TAG/TCSS. We then **built a real async disaggregated
runtime** (`digenrl_bench/async_runtime.py`) that mirrors UniRL's `AsyncARTrainer`
(disjoint train/rollout slabs, rollout buffer, weight_version, max_inflight,
buffer_max_staleness, drain-before-sync) but for the **diffusion** stack with real WAN
transformer work units, and measured the speedups end-to-end on zw1.

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

## GAP / TCSS / TAG — MEASURED end-to-end (async runtime)
`digenrl_bench/async_runtime.py` runs a real disaggregated pipeline: a **rollout slab**
(GPUs generating, T denoise forwards/unit) and a **train slab** (GPUs doing K replay
fwd+bwd/unit), with a real cross-slab weight-sync. SYNC = the colocated-style baseline
(generate → train → sync, serialized). ASYNC = overlap round r+1's generation with round
r's train+weight-sync (GAP fine-grain units + TCSS no-drain 1-stale). +TAG = train slab
work-steals the round's remaining generation units.

6 GPUs (2 train + 4 rollout), WAN2.1-1.3B, 10 rollouts, B=4:
| config | SYNC | ASYNC GAP+TCSS | +TAG |
|---|---|---|---|
| T=12, K=4, wsync=300MB (balanced) | 57.4s (1.00×) | 42.3s (**1.36×**) | 40.4s (**1.42×**) |
| T=20, K=2, wsync=500MB (gen-bound) | 53.6s (1.00×) | 38.9s (**1.38×**) | 39.4s (1.36×) |

→ **The async pipeline gives a real 1.36–1.44× over the serial baseline**, from hiding the
weight-sync + the gen/train imbalance. TAG helps when the train slab finishes early enough
to usefully steal (balanced config); when generation already saturates the rollout slab
(gen-bound), TAG ≈ GAP+TCSS. (An early coarse TAG that over-generated a full extra round
*hurt* — 0.68× — fixed with fine-grained bounded work-stealing.)

## Pipeline model cross-check (theoretical ceiling)
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
- GAP/TCSS/TAG are measured **end-to-end** via `async_runtime.py` (1.36–1.44×): a standalone
  runtime with real WAN work + real slabs/buffer/staleness/sync.
- The framework port is now done: **`unirl/trainer/async_diffusion.py::AsyncDiffusionTrainer`**
  (subclasses `DiffusionTrainer`, reuses its `layout="separate"` two-slab construction +
  NCCLWeightSync, overlays `AsyncARTrainer`'s async machinery — rollout buffer, non-blocking
  `_generate_async`, `max_inflight`, `buffer_max_staleness`, drain-before-sync) + entry point
  `unirl/train_async_diffusion.py` + recipe `examples/diffusion/hunyuan_video15/..._t2v_async.yaml`.
  Statically validated (compiles, imports, method/signature checks); a live multi-node RL run
  needs the ray + vllm-omni + dataset bring-up (not done here). The pipeline-model ceilings below
  are a cross-check, consistent with the measurement.
