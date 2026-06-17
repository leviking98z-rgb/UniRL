# Partial rollout (budget-resume generation)

## The problem it addresses

In RL rollout, generation length is heavy-tailed: most sequences finish quickly
but a few run to the `max_new_tokens` cap. A single `generate` call must wait for
the slowest sequence (the *straggler*), so its wall-clock and its peak KV-cache
footprint are both set by the tail, not the median.

**Partial rollout** caps how many tokens a single engine round may emit. A
sequence that does not finish within the cap is *resumed* on the next round with
`input_ids = prompt + tokens-generated-so-far`. Finished sequences drop out (their
KV frees immediately). This bounds per-round time and per-round KV by a fixed
**budget** instead of by the straggler.

## Design

One config field — `rollout.config.partial_budget` (`SGLangEngineConfig`,
default `0` = off, one-shot generation). When `> 0`, `SGLangRolloutEngine.generate`
delegates to `_generate_partial(prepared, budget, max_total)`:

1. Each round, build a sub-batch of the still-active sequences with
   `max_new_tokens = min(budget, max_total - len(generated))`.
2. Call `backend.generate` on the sub-batch (one bounded round).
3. Accumulate `token_ids` and `logprobs` per sequence. A sequence is done on
   `finish_reason in {stop, eos, matched_stop}`, on reaching `max_total`, or when
   the round returns no tokens.
4. Loop until no active sequences remain, then decode the full text and hand one
   accumulated result per sequence to `adapter.build_response` — the same typed
   `RolloutTrack` the one-shot path produces. **The resume loop runs at the raw
   token/logprob level; the typed batch is built only once, at completion**, so
   the framework's CONCAT-field `Segment`/`Batch` types never see a partial state.

### Correctness

Within one `generate` the policy weights do not change, so the per-token logprobs
recorded each round (`return_logprob`) are the true behaviour-policy logprobs and
concatenate cleanly across resumes — no replay needed. Across weight versions
(when paired with the async trainer), the off-policy gap is absorbed by DRPO's
rollout-anchored ratio (`algorithm.old_logp_source: rollout`), exactly as the
async buffer already relies on.

## Usage

```bash
# colocate sync trainer
ENTRY=train_ar bash examples/run_experiment_single_node.sh \
  ar/qwen3_drpo_4b_base_dapo_sglang \
  +rollout.config.partial_budget=256

# async trainer (#68), v2 sglang engine
ENTRY=train_async_ar bash examples/run_experiment_single_node.sh \
  ar/qwen3_drpo_4b_base_dapo_sglang_async \
  +rollout.config.partial_budget=256
```

## When it helps — and when it costs

Partial rollout is **not** a free speedup. It does the same total token work split
into more, smaller rounds, and each resume re-prefills the grown prompt prefix
(radix cache mitigates this only when the prefix survives in the worker's cache).

| Regime | Effect |
|---|---|
| KV-bound (high concurrency / long `max_new_tokens` / small `mem_fraction_static`) | **Win** — finished sequences free KV every round, raising the concurrency ceiling and avoiding KV-overflow eviction. |
| Extreme length variance + limited generation overlap | **Win** — bounds the per-round straggler cost. |
| Small batch, ample KV, async overlap already hiding stragglers | **Overhead** (~15% in our measurement) — the extra rounds and re-prefill have no headroom to reclaim. |

The async trainer (`#68`) already hides stragglers at *generation* granularity
(non-blocking futures + `max_inflight` overlap + a staleness-bounded group buffer),
so partial rollout's marginal value there is the per-round KV-freeing and time
bound, realised only under memory pressure.

## Validation (Qwen3-4B-Base, DAPO-Math, 8xGPU)

- **Colocate sync DRPO, `partial_budget=256`, `max_new_tokens=2048`**: runs
  end-to-end; `reward_mean ~ 0.13` (matches the one-shot baseline), `grad_norm`
  healthy, active-request count decays as sequences finish. Correct + trainable.
- **Async DRPO (`train_fraction=0.5`, `max_inflight=2`, `buffer_max_staleness=1`),
  `partial_budget` 0 vs 256**: both `EXIT=0`, `ratio_mean ~ 1.00` (NCCL cross-slab
  weight sync correct, on-policy), `reward_mean` sane. `perf/rollout_time_s`:
  `0 -> 8.08s`, `256 -> 9.34s` (**+15%** — the no-headroom overhead case above).

## Note on the async recipes

`examples/ar/qwen3_*_async.yaml` previously targeted the retired
`sglang_llm` engine (deleted with the v2 engine rewrite). They are migrated here
to `unirl.rollout.engine.sglang.engine.SGLangRolloutEngine` /
`SGLangEngineConfig`; the v2 HTTP backend serves the NCCL distributed
weight-update endpoints the disaggregated `NCCLWeightSync` drives.

## Follow-up: where the resume cost actually comes from (measured)

A controlled probe (standalone SGLang server, no weight sync, sequential
re-submit of `prompt + generated-so-far`) shows the resume is **nearly free** when
the radix/prefix cache survives:

```
resume round 1: prefix_in=351  cached-token=350  new-token=1
resume round 2: prefix_in=607  cached-token=606  new-token=1
resume round 3: prefix_in=863  cached-token=862  new-token=1
```

i.e. the engine re-prefills ~1 token, not the whole prefix — radix matches the
already-computed prefix. So **re-submission is not inherently expensive**; the
earlier +15% was an artifact, not the mechanism.

The artifact has a concrete cause: the rollout engine **flushes the KV/radix cache
by design** — `SGLangRolloutEngine.sleep()` flushes before releasing memory, and
every weight sync flushes (`flush_cache=True`). With `weight_sync_interval=1` the
cache is wiped each step, so any resume that spans a sleep/sync boundary re-prefills
from scratch (observed `cached-token=0` across all engine-path resumes). KV headroom
also matters: at `mem_fraction_static=0.3` (colocate default) the pool evicts under
concurrency.

Measured net overhead of `partial_budget=256` vs one-shot, same hardware:
- async (`mem_fraction=0.8`, `weight_sync_interval=1`): rollout time +15%
- colocate (`mem_fraction=0.6`): rollout time +5.8% (66.2s -> 70.0s), reward parity

So the cost is **modest and headroom/flush-dependent**, not catastrophic. The path
to slime/veRL-level near-zero overhead is to keep a sequence's resume rounds inside
a single non-flushed, KV-retained window — i.e. interrupt only the straggler tail at
the sync boundary (slime `--partial-rollout`) rather than chopping every sequence
into budget rounds, and avoid flushing the radix cache between a sequence's resumes.

## Implemented: slime-style interrupt-at-sync + carry (AsyncARTrainer)

Enabled with `+partial_rollout=true` on the async recipe. Instead of the
pre-sync `_drain_all()` blocking on the straggler, the trainer:

1. **Aborts** all in-flight generations — the driver POSTs `/abort_request`
   directly to each SRT server (the ray engine actor is busy inside generate),
   so each pending `generate()` returns its partial tokens+logprobs with
   `finish_reason=abort`.
2. **Carries** any generation that has interrupted samples: its accumulated
   per-sample tokens/logprobs/text are hydrated to concrete tensors (so they
   survive the sync) and stashed whole (whole generations → GRPO groups never
   split). Complete generations score+buffer as usual.
3. Weight sync runs (no straggler wait).
4. **Continues** each carried generation under the new weights: a continuation
   req (`input_ids = prompt + tokens-so-far`, remaining-budget clamped, padded
   to the rollout DP size for DP_SCATTER) regenerates only the unfinished
   samples; `merge_continuation` folds the new tokens back. A sequence may span
   several weight versions; off-policy is absorbed by `old_logp_source=rollout`
   (recorded behaviour logprobs), exactly as the staleness buffer already does.

Pieces: `finish_reason` plumbed into `RolloutTrack` (adapter `build_response`);
`RolloutReq.continuation_token_ids` + `build_inputs` append/clamp; pure
carry/merge helpers in `unirl/trainer/_partial_rollout.py` (unit-tested);
orchestration (`_abort_servers`/`_ingest_partial`/`_abort_and_carry`/
`_relaunch_carry`) in `AsyncARTrainer`, guarded by `partial_rollout` (default
off → byte-identical to the existing async path).

### e2e validation (Qwen3-4B, DAPO, 8xH20, train_fraction=0.5, max_inflight=3, weight_sync_interval=1, FORCE_IGNORE_EOS)
EXIT=0 over 6 rollouts; **6 sync boundaries** exercised carry+relaunch across
weight versions (e.g. "1 carried, relaunched under weight v1/v2"); `reward_mean`
≈0.125 (DRPO early baseline), `ratio_mean`≈0.9996 (off-policy carry does NOT
break the importance ratio). Helper logic covered by GPU-free unit tests
(concat/split/balance_shards preserve finish_reasons; carry/merge/complete +
logprob alignment).

## Sync-barrier saving + the repeated-abort optimization (measured)

With partial rollout the weight-sync barrier no longer drains the straggler — it
aborts and carries. But a naive ONE-SHOT abort does **not** save time: with
`max_inflight>1`, each in-flight generation's HTTP backend (asyncio.gather,
`concurrency`) keeps submitting its queued requests *after* the abort fires, so
those late arrivals run to completion and gate the collect (`ray.get` waits for
every per-worker generate). Measured: one-shot-abort quiesce ≈ baseline drain
(~40s); `abort_post=0.1s` but `collect=34–92s`; the SRT `#running-req` drops
16→1–3 on abort (it does interrupt running decode) but the residual late-submitted
requests dominate.

**Optimization** (commit `c560217`): re-POST `abort_all` every 0.3s from a daemon
thread for the whole duration of the collect, catching requests as the backends
submit them. This drops the barrier quiesce to ~7s.

### Measured impact (Qwen3-4B, DAPO, 8×H20, train_fraction=0.5, max_inflight=3, weight_sync_interval=1, natural heavy-tail generation, 50 rollouts each)

| | baseline (drain) | partial (repeated-abort) |
|---|---|---|
| reward mean (50 rollouts) | 0.211 | 0.208 |
| reward windows of 10 | 0.153 → 0.222 | 0.136 → 0.292 |
| sync-barrier quiesce | 37.3s | **7.2s** (−81%) |
| per-rollout wall-clock | 47.0s | **17.7s** (**2.65× throughput**) |
| ratio_mean | ~1.0 | 0.998–1.0005 |

**Reward converges identically** (0.211 vs 0.208; both curves rise and track) — the
off-policy carry costs nothing here — while throughput is **2.65×**. Caveat: the
speedup is maximal at `weight_sync_interval=1` (a barrier every rollout); it scales
down as syncs become less frequent (fewer barriers to save). sglang's
`abort_request` does interrupt running decode (`to_finish=FINISH_ABORT`), so the
saving is real, not a generation-length artifact.

## IMPORTANT: throughput benefit is regime-dependent (the 2.65× is NOT universal)

The 2.65× above was measured at **batch_size=8, max_new_tokens=2048** — a regime
where the per-rollout cycle is short and the sync-barrier *drain* (waiting for the
in-flight straggler) dominates per-rollout time (drain 37s of 47s). There, removing
the barrier is a 2.65× win.

A second bench at **batch_size=64, max_new_tokens=8192** (matching the optstack
vanilla-GRPO reference run b31s0usr, 512 samples/rollout) shows **no throughput
gain**:

| | baseline (drain) | partial (repeated-abort) |
|---|---|---|
| per-rollout wall | 128.1s | 126.0s (~1.0×) |
| sync-barrier | drain mean 8.2s | abort 0.3s |
| reward (convergence) | tracks b31s0usr | tracks b31s0usr (0.13→0.33, ratio≈1) |

Why: at batch_size=64 the run is **generation-throughput-bound** (~126s of actual
gen+train per rollout). With , by the time a sync fires the
in-flight generations have been running ~2 rollout-cycles and are nearly complete,
so the drain is only ~8s — a small fraction of 126s. Aborting it saves ~8s →
negligible.

**Rule of thumb**: partial rollout's throughput benefit ≈ (sync-barrier drain) /
(per-rollout wall). It is large when the drain dominates (small batch, short cycle
relative to generation time) and ~zero when the run is generation-bound (large
batch). In **both** regimes partial **converges identically and never hurts**
(126 ≤ 128s; reward aligned) — it is a safe default whose speedup is opportunistic.
The carried-straggler compute is deferred, not eliminated, so total generation work
is unchanged; the win (when present) is purely from not stalling the train pipeline
at the barrier.
