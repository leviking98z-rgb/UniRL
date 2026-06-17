# FP8 rollout (low-precision generation, BF16 training)

## The problem it addresses

In disaggregated async RL (`AsyncARTrainer`), the rollout engine sits on its own
GPU slab and stays resident; generation throughput sets how fast the training
slab is fed. SGLang can run **generation** in FP8 (FP8 weights + activations,
optionally an FP8 KV cache), which roughly halves the rollout engine's memory
footprint and speeds decode — without touching the training math.

This is sound because **the policy being optimized is the BF16 train weights**,
not the rollout engine. The rollout engine is only the *sampler* — it draws
trajectories from the behaviour policy `mu`. Running that sampler in FP8 changes
*which* trajectories get drawn (and the `mu` logprobs it reports), but the
gradient is still computed against the BF16 forward. Rollout precision and train
precision are decoupled by construction.

## The mechanism / config flag

One typed field on `SGLangEngineConfig` (`unirl/rollout/engine/sglang/config.py`):

```yaml
rollout:
  config:
    _target_: unirl.rollout.engine.sglang.config.SGLangEngineConfig
    quantization: fp8          # default None = native dtype (current bf16, unchanged)
    kv_cache_dtype: fp8_e5m2   # optional; default None = ServerArgs default
```

`server_intent()` emits `quantization` / `kv_cache_dtype` into the ServerArgs
intent **only when set**, so a recipe that omits them produces a byte-identical
intent to the pre-FP8 code path. The HTTP/native backends already filter the
intent against the live `ServerArgs` fields at boot, so the keys land on
`ServerArgs.quantization` / `ServerArgs.kv_cache_dtype` by name — no per-backend
plumbing. `__post_init__` validates `quantization` against SGLang's known choices
(`fp8`, `w8a8_fp8`, `awq`, ...) so a typo fails at config-build time instead of as
an opaque error inside the SRT subprocess.

The training side is untouched: FSDP `param_dtype`, `autocast_precision`, and
`logprob_precision` all stay exactly as the recipe sets them (bf16 forward, fp32
logprobs). No knob bleeds from rollout precision into the optimizer.

## The correctness argument (the IS correction it relies on)

**HAZARD.** A teacher-forced replay of an FP8-generated trajectory through the
BF16 train forward diverges from the FP8 engine's own logprobs **more** than
bf16-vs-bf16 does. If that gap were ignored, the update would be biased: we would
weight gradients by an `mu` the engine never actually sampled from.

It is *not* ignored. The AR algorithms anchor the importance ratio on the
**rollout engine's emitted logprobs** (`algorithm.old_logp_source='rollout'`, the
default for GRPO/CPPO/DRPO). The per-token ratio is

```
r_t = exp(new_logp - old_logp) = pi_BF16(a_t) / mu_FP8(a_t)
```

where `old_logp = mu_FP8` is what the FP8 engine reported and `new_logp =
pi_BF16` is the train forward. This is exactly an **importance-sampling
correction**: the surrogate `-A_t * r_t` is unbiased for the BF16 policy's
objective *as long as the ratio is well-behaved* — the FP8↔BF16 gap is corrected,
not swept under the rug. Concretely, FP8 rollout reuses the same off-policy
machinery that already absorbs the async staleness gap (`buffer_max_staleness >
0`); FP8 just adds an engine-precision term to the same ratio.

The AR algorithms already emit `rollout_replay_logp_absdiff_mean` — the mean
per-token `|Δlogp|` between rollout and replay (`unirl/algorithms/base.py`). This
is the direct, symmetric gauge of the FP8↔BF16 gap.

### The guard — do not silently train through a large mismatch

IS is only unbiased when the ratio has a **light tail**. FP8 can be aggressive
enough (certain layers, certain models) that the tail is heavy — then the
estimator's variance explodes and the update destabilizes. `AsyncARTrainer` reads
`rollout_replay_logp_absdiff_mean` off each `TrainStepResult` and applies two
opt-in thresholds (`unirl/trainer/async_ar.py:_guard_rollout_drift`):

```yaml
rollout_drift_warn:  0.10   # log a warning past here (tail widening; keep watching)
rollout_drift_abort: 0.30   # RuntimeError past here (refuse to train through it)
```

Both default to `None` = off, so a bf16-rollout recipe is unaffected. With FP8
enabled, set them so a runaway gap surfaces loudly instead of corrupting the run.
The numbers above are starting points; on-policy bf16-vs-bf16 typically sits well
under `0.05`, so a sustained `> 0.1` mean already says the FP8 engine and the
BF16 forward disagree materially.

### TODO — Truncated Importance Sampling (TIS)

The current guard is binary (warn / abort). The principled next step when the
tail is heavy but bounded is **Truncated Importance Sampling**: clamp the ratio
at a cap `c` in the AR loss,

```
r_t  <-  min(r_t, c)
```

which trades a small, controlled bias for a large variance reduction and keeps
the FP8 run training instead of aborting. The hook is marked in
`AsyncARTrainer._guard_rollout_drift`; the clamp itself belongs in the per-token
AR losses (`unirl/algorithms/cppo.py::_cppo_loss`,
`unirl/algorithms/drpo.py::_drpo_loss`, where `ratio = exp(log_diff)` is formed).
CPPO's Binary-TV mask and DRPO's smooth χ² regularizer already attenuate large
`|r_t - 1|`, so TIS is complementary, not redundant — it caps the *gradient
weight* the ratio contributes before the trust region sees it.

## MoE caveat (for the future)

For Mixture-of-Experts models (e.g. Qwen3-30B-A3B), FP8 the **expert** weights
but keep the **router/gating** in higher precision. The router's top-k argmax is
discontinuous: an FP8 rounding error in the gate logits can flip which experts a
token is routed to, which changes the *support* of `mu` — not just its density.
That is a far worse rollout↔replay divergence than dense FP8 (the BF16 replay may
route the token to a different expert entirely, so `new_logp` and `old_logp`
describe different computations). The fix is two-fold:

1. **Router-higher-precision** — keep the gate in bf16/fp16 (SGLang exposes this
   via its quantization config; a future field would surface it here) so the
   routing decision matches between FP8 generation and BF16 replay.
2. **Router-replay** — record the engine's per-token expert routing during
   generation and replay it on the train side, so the BF16 forward is forced
   through the *same* experts the FP8 engine used. This makes the ratio a true
   per-token IS ratio again rather than a comparison across different subnetworks.

Neither is implemented here; dense FP8 rollout is the scope of this change. The
drift guard above is the safety net that will catch an MoE run whose routing
diverges before it silently corrupts training.
