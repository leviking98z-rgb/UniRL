#!/usr/bin/env python
"""DigenRL per-technique speedup model, driven by MEASURED primitives.

TSP is measured directly (replay seq vs batched). GAP / TAG / TCSS are
disaggregated-pipeline techniques whose speedup depends on the generator vs
trainer time balance — we compute them from the measured per-step costs using
DigenRL's bubble model. Assumptions are printed explicitly.

Measured on zw1 (8-GPU FSDP, WAN2.1-1.3B, latent [1,16,5,30,52]):
  gen_step  = 130 ms   (one denoise-step forward, no-grad, CFG)  -> generator
  train_step= 380 ms   (one selected-step replay fwd+bwd)         -> trainer
  TSP       = 1.07x    (K=2..4, replay forward batching)
"""

GEN_STEP_MS = 130.0     # measured: generator per denoise step
TRAIN_STEP_MS = 380.0   # measured: trainer per selected step (fwd+bwd)
TSP_SPEEDUP = 1.07      # measured: replay forward-batching

def analyze(T, K, M, tsp=False):
    """T = denoise steps (generation), K = trained steps, M = pipeline micro-batches."""
    gen = T * GEN_STEP_MS
    train = K * TRAIN_STEP_MS * (1.0 / TSP_SPEEDUP if tsp else 1.0)

    colocated = gen + train                       # baseline: serialize on one pool (UniRL today)
    # GAP: micro-pipeline overlap; fill+drain bubble ~ (gen+train)/M, shrinks as M grows.
    #   GAP's role is enabling large M (fine micro-batches) under small diffusion batches.
    gap = max(gen, train) + (gen + train) / M
    # TCSS: 1-step async removes the residual fill/drain bubble -> the pure-overlap ceiling.
    tcss = max(gen, train)
    # TAG: idle group (when gen!=train) does the other stage's work -> upper bound is perfect
    #   work-stealing where all GPUs always busy on the combined load = (gen+train)/2.
    #   (Realistic gain sits between TCSS and this bound; the split/fungibility isn't measured.)
    tag = max((gen + train) / 2.0, min(gen, train))   # can't beat the smaller irreducible stage

    return {
        "gen_ms": gen, "train_ms": train,
        "colocated_ms": colocated,
        "GAP_ms": gap, "GAP_speedup": colocated / gap,
        "TCSS_ms": tcss, "TCSS_speedup": colocated / tcss,
        "TAG_ms": tag, "TAG_speedup_upper": colocated / tag,
        "overlap_ceiling": colocated / max(gen, train),
    }

def show(label, T, K, M, tsp):
    r = analyze(T, K, M, tsp)
    print(f"\n=== {label}: T={T} denoise, K={K} trained, M={M} micros, TSP={'on' if tsp else 'off'} ===")
    print(f"  generator total = {r['gen_ms']/1000:.2f}s   trainer total = {r['train_ms']/1000:.2f}s")
    print(f"  colocated (baseline)        {r['colocated_ms']/1000:6.2f}s   1.00x")
    print(f"  + GAP  (micro-pipeline)     {r['GAP_ms']/1000:6.2f}s   {r['GAP_speedup']:.2f}x")
    print(f"  + TCSS (1-step async)       {r['TCSS_ms']/1000:6.2f}s   {r['TCSS_speedup']:.2f}x  (= overlap ceiling)")
    print(f"  + TAG  (work-steal, upper)  {r['TAG_ms']/1000:6.2f}s   {r['TAG_speedup_upper']:.2f}x  (upper bound)")

if __name__ == "__main__":
    print("DigenRL per-technique speedup (from measured gen=130ms/step, train=380ms/step, TSP=1.07x)")
    print("Each row is cumulative wall-clock per RL step under that technique vs colocated baseline.")
    # representative configs (T denoise steps, K trained steps, M micro-batches)
    show("balanced (K~T/3, gen-bound)", T=50, K=16, M=8, tsp=True)
    show("train-heavy (K=T, train-bound)", T=50, K=50, M=8, tsp=True)
    show("rollout-heavy (K=T/5)", T=50, K=10, M=8, tsp=True)
    print("\nTSP (measured, standalone, replay forward-batching): 1.07x on 8-GPU FSDP WAN2.1-1.3B")
