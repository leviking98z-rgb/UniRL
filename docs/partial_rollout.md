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
