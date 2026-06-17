# Decoupled reward (off-critical-path scoring)

## The problem it addresses

After a generation completes, the async AR trainer scores it before the group
can enter the rollout buffer:

```
reap generate → reward.score_and_attach (BLOCKS the driver) → buffer → train
```

`score_and_attach` is a `@distributed(DP_SCATTER)` Handle call that blocks the
single driver thread inside `ray.get` until every reward shard returns
(`unirl/distributed/group/handle.py`). For an in-process verifier (math-verify,
a small CPU/GPU scorer) that is microseconds and the block is irrelevant. For a
**heavy verifier** — a code-execution sandbox, an LLM-as-judge, a remote reward
server (`unirl.reward.remote`) — that block can be seconds, and it sits on the
critical path: the next generation and the next train step wait behind it.

UniRL already overlaps *generation* with training (non-blocking Ray futures +
`max_inflight` + the staleness-bounded buffer in `AsyncARTrainer`), so each
group's generation is hidden. **Reward** is the one remaining synchronous hop.

## What overlaps

When enabled, the blocking `score_and_attach` runs in a bounded thread pool
instead of on the driver thread, so it overlaps the next `generate`/`train`
step:

```
reap generate → SUBMIT scoring to pool ──┐ (driver returns immediately)
                                         │  (scoring runs concurrently)
launch next generate / run train step    │
reap scoring (when done) → buffer → train ┘
```

The dispatch itself is unchanged — the same `RewardService.score_and_attach`
runs the same DP-sharded reward. Only *where it blocks* moves: off the driver,
onto a worker thread that releases the GIL inside `ray.get`. This is the same
launch/reap split the trainer already uses for `generate`
(`_generate_async`/`_collect_resp`), applied to reward.

## Design

`_DecoupledScorer` (`unirl/trainer/async_ar.py`) is a `ThreadPoolExecutor`
keyed by `gen_id`:

- **`submit(gen_id, score_fn, on_done)`** — launch `score_fn` (the blocking
  `score_and_attach` for the generation's scorable tracks) in the pool. The
  driver does not block on it. `on_done` is the buffer step, run later on the
  driver thread.
- **`reap_ready()`** — called from `_reap_ready`: harvest every scoring that has
  finished and run its `on_done` (drop decoded + `track.split()` +
  `buffer.put`) **on the driver thread**, so the rollout buffer stays
  single-threaded and lock-free.
- **`drain()`** — block until every in-flight scoring is reaped + buffered.
  Called from `_drain_all` and the partial-rollout `_abort_and_carry`, i.e. the
  mandatory barriers before a weight sync, eval, or checkpoint.

The driver-side seam is `_score_into_buffer`, split into:

- `_score_tracks(rec, resp)` — the blocking reward call; pure (no driver-state
  mutation), so it is safe to run in a worker thread.
- `_finish_into_buffer(rec, resp, scored)` — attach scored tracks, drop decoded,
  split into groups, `buffer.put`; driver-thread only.

Default off (`reward_decoupled=false`) runs both inline, exactly as before.

### Bounded concurrency + timeout

- **`reward_max_concurrent_scorings`** (default `2`) — the pool width = max
  in-flight scorings. `submit` blocks once the cap is reached (reaping one
  first), so a fast-generating / slow-scoring run applies natural backpressure
  instead of fanning out unbounded scoring tasks.
- **`reward_score_timeout_s`** (default `null` = wait indefinitely, today's
  behavior) — per-scoring wall-clock cap for slow verifiers. On timeout the
  future raises `TimeoutError` at reap, **failing the run** rather than silently
  dropping a group (same fail-fast posture as the inline path's per-sample
  failure flags). Set it for code-exec / judge rewards that can hang. "Adaptive"
  timeouts (scale with batch size / observed latency) are a natural extension of
  this single knob; the cap is the building block.

## Correctness

The decoupled path preserves every invariant of the inline path:

- **Every group is scored before it is trained on.** Groups only enter the
  buffer via `_finish_into_buffer`, which runs *after* `score_and_attach`
  returns. `_next_batch` draws exclusively from the buffer, so an unscored group
  can never be selected for training.
- **The weight-sync / eval / checkpoint barriers are honored.** `_drain_all`
  (and `_abort_and_carry` under partial rollout) call `_scorer.drain()` after
  collecting in-flight generations, so no scoring is in flight when weights
  update — preserving the on-policy / staleness-bounded semantics and avoiding
  any KV/weight corruption window.
- **Ordering / identity preserved.** Scorings are keyed by `gen_id`; each
  group is stamped with its `(weight_version, gen_id)` exactly as before, so the
  staleness buffer's freshness ordering is unchanged.
- **Single-threaded buffer.** Only the blocking reward call runs in a thread;
  all `_buffer`/`resp` mutation stays on the driver thread at reap time, so no
  lock is introduced.
- **Fail-fast unchanged.** A scoring exception (including a verifier failure
  flag or a timeout) propagates from `future.result()` at reap and aborts the
  run, same as the inline `raise`.

## Usage

Top-level knobs on the async recipe (the `reward:` block is reserved for
`RewardService.__init__` kwargs, so these live alongside the other async knobs):

```bash
ENTRY=train_async_ar bash examples/run_experiment_single_node.sh \
  ar/qwen3_grpo_4b_base_dapo_sglang_async \
  +reward_decoupled=true \
  +reward_max_concurrent_scorings=4 \
  +reward_score_timeout_s=120
```

Default (`reward_decoupled` unset / `false`) is byte-identical to the existing
inline per-group scoring.

## When it pays — and when it doesn't

| Regime | Effect |
|---|---|
| Heavy verifier (code-exec sandbox, LLM-judge, remote reward server) | **Win** — seconds of scoring per group overlap the next generate/train instead of stalling the driver. |
| Fast in-process scorer (math-verify, CPU check, small reward model) | **No measurable gain** — the inline block was already negligible; decoupling only adds thread-pool bookkeeping. Keep it off. |
| Scoring far slower than a generate+train cycle | **Backpressure-bound** — `reward_max_concurrent_scorings` caps overlap; the run is then reward-throughput-bound and the right fix is a bigger reward resource pool / server-RM (below), not a larger thread pool. |

Because UniRL already overlaps generation per group, decoupled reward's value is
concentrated in the **heavy-verifier** case. It is a safe default-off opt-in:
when the reward is cheap it does nothing useful, so only enable it when the
verifier is slow.

## Upgrade path: resource pool / server-RM

The thread pool is the smallest unit that removes the driver block; it does not
add reward *throughput* (the underlying `RewardService` still does the same
work). When scoring is the bottleneck, the path forward is to scale the reward
backend itself, reusing the same `submit`/`reap`/`drain` seam:

1. **Reward resource pool** — back the scorer with multiple `RewardService`
   replicas (a Ray actor pool) and round-robin `submit` across them, so N
   verifiers run truly in parallel. The thread pool's bounded-concurrency +
   timeout + driver-side reap stay; only `score_fn` changes to target a pooled
   replica.
2. **Server-RM** — point `RewardService` at the remote HTTP reward server
   (`unirl.reward.remote`, already supported as a backend) and let the server
   own batching / autoscaling across many judge or sandbox workers. Decoupling
   then matters most, because a single remote round-trip is the slow hop being
   hidden, and the server amortizes load across requests in flight.

Both upgrades are additive on top of this change: the trainer-side contract
(launch off the critical path, reap on the driver, drain at every barrier) is
unchanged.
