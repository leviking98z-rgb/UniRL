# In-flight continuous weight update (PipelineRL-style)

Roadmap issue #94, item #6.

## The problem it addresses

In the async AR trainer (`AsyncARTrainer`, #68) training and SGLang rollout run on
disjoint GPU slabs and generation overlaps training. But at each weight sync the
engine must **quiesce**: every in-flight `generate` is finished or interrupted
before the trainer pushes new weights, because the default
`NCCLWeightSync.sync()` ends with `flush_cache=True` on the last bucket — flushing
the SRT KV/radix cache mid-decode would drop the running sequences' KV and
corrupt them. That quiesce is the *sync barrier*:

- baseline (`_drain_all`): block until the in-flight straggler finishes, then sync;
- partial rollout (#2, `partial_rollout=true`): abort the in-flight requests,
  carry the partial tokens, sync, then **relaunch** the carried sequences as fresh
  `generate` calls under the new weights.

Both stop generation at the barrier. The straggler tail still gates throughput
(baseline) or pays an abort + re-prefill + relaunch round-trip (partial).

**In-flight continuous weight update** removes the barrier entirely. The trainer
pushes the new weights into the *live* engine **without flushing the KV cache and
without stopping any request**. A sequence that is mid-generation simply keeps
decoding: its prefix KV stays valid, and from its *next* token onward it decodes
under the new weights. Engines never quiesce; there is no abort, no drain, no
relaunch.

This is the PipelineRL design: the inference server is a continuously-running
producer, and weight updates are applied to it as a side-effect between decode
steps rather than at a stop-the-world barrier.

## Mechanism

One config flag — `in_flight_weight_update` (default `False`). When off, the trainer
is **byte-for-byte the existing async path** (drain-or-abort barrier, flushing
sync). When on, the per-sync block in `AsyncARTrainer.train` becomes:

```python
# in-flight: push weights into the LIVE engine, no quiesce barrier.
self.weight_sync.sync(flush_cache=False)
self._weight_version += 1
```

Two pieces make this work:

1. **`NCCLWeightSync.sync(flush_cache=False)`** — a new optional override on the
   existing `sync()`. The default (`flush_cache=None` ⇒ the handler's
   `self._flush_cache`, flushed on the last bucket) is unchanged for the quiesced
   paths. With `flush_cache=False` the broadcast still runs the normal train-mesh
   all-gather and the NCCL bucket loop, and the receiver's
   `update_weights_from_distributed` still swaps the model weights on the live
   scheduler — sglang applies a distributed weight update between decode steps,
   it does **not** require the scheduler to be idle. Only the KV/radix flush is
   suppressed, so running sequences keep their KV.

2. **No barrier in the trainer.** The `in_flight` branch calls neither
   `_drain_all()` (baseline) nor `_abort_and_carry()` + `_relaunch_carry()`
   (partial). The in-flight `generate` futures are left running across the sync;
   `_reap_ready` harvests them whenever they complete, exactly as in steady state.

`save`/`eval` still drain first (they share the engine and need a consistent,
quiesced model), and the `finally` still drains — those are unchanged.

## Per-token weight-version correctness

The danger of removing the barrier is that **a single sequence may now span
multiple weight versions token-by-token**: tokens before the sync were produced by
version *V*, tokens after by *V+1* (and a long sequence may cross several syncs).
For the policy-gradient update to be valid this must not break the importance
ratio.

It does not, and the reason is the recorded behaviour-policy logprob. With
`algorithm.old_logp_source: rollout` (the canonical async/DRPO setting, already
required by partial rollout and the staleness buffer) the *old* (behaviour) logprob
in the ratio `π_θ(a|s) / π_behaviour(a|s)` is **the logprob the engine emitted at
generation time**, carried per token on the segment:

- SGLang generates with `return_logprob`; the adapter's `build_response` packs
  `log_probs = [r.logprobs ...]` straight from the SRT raw result
  (`unirl/rollout/engine/sglang/adapters/text.py`). Each entry is the logprob the
  engine actually used to sample that token, **computed against whatever weight
  version was live when that token was decoded**.
- So when a sequence crosses a sync, the tokens decoded under *V* carry *V*'s
  logprobs and the tokens decoded under *V+1* carry *V+1*'s logprobs — automatically,
  with no extra tagging. The per-token behaviour logprob *is* the per-token weight
  version, expressed in the only quantity the ratio needs.
- The new-policy logprob `π_θ` is recomputed on the train side over the full token
  sequence; the ratio is then exact per token, and the within-sequence version mix
  is absorbed token-by-token. This is the same property partial rollout relies on
  to fold a carried sequence's pre- and post-sync halves — here it just happens
  continuously instead of at carry/merge time.

### What is guaranteed, and what is not

**Guaranteed:**

- The per-token behaviour logprob recorded on the segment is the one from the
  weight version that produced that token (it is the engine's own emission at decode
  time). The ratio is therefore valid token-by-token even when a sequence spans
  versions. This is the load-bearing correctness property and it is exact.
- Weight-version bookkeeping stays monotonic and conservative. Each buffered GRPO
  group is stamped with its **launch** weight version — the oldest weights any token
  in it could have used — so `buffer_max_staleness` eviction keys on the most-stale
  bound (a group is treated as at least as stale as its earliest token). No group is
  admitted as fresher than it really is.

**Not guaranteed (documented limitation):**

- There is **no explicit per-token integer version tag** on the segment. The version
  identity lives implicitly in the recorded per-token logprob (which is sufficient
  for the ratio). A separate `int` weight-version-per-token array would be the
  fully-explicit form; it is **not** implemented in this commit because it is
  invasive (it requires threading a version vector through the SRT raw result, the
  adapter `build_segment` pack, the `Segment`/`Batch` CONCAT plumbing, and every
  consumer) and the ratio does not need it. The staleness *accounting* therefore
  uses the conservative launch-version bound above rather than a per-token
  histogram. If a future change wants per-token staleness weighting (e.g. a
  version-decayed loss), that explicit tag is the follow-up.
- A sequence whose KV was written under *V* and continues under *V+1* is a genuine
  hybrid — its early KV reflects *V*'s attention/values, not recomputed under *V+1*.
  This is intrinsic to "keep decoding without recompute" and is exactly the
  PipelineRL approximation; it is the same hybrid that partial rollout produces when
  it continues a carried prefix, except here the prefix KV is *reused in place*
  rather than re-prefilled. The recorded logprobs still describe the actual sampling
  distribution, so the ratio is unaffected; the only thing not "as if regenerated
  from scratch under V+1" is the hidden KV, which RL never observes directly.

## How it differs from partial rollout (#2)

Both remove the sync barrier, but by opposite means, and they are **mutually
exclusive** (the trainer raises if both are set):

| | partial rollout (#2) | in-flight weight update (#94) |
|---|---|---|
| at the sync barrier | **abort** all in-flight requests | **nothing stops** |
| running sequences | interrupted (`finish_reason=abort`), carried | keep decoding uninterrupted |
| after the sync | **relaunch** carried seqs as fresh `generate` (re-prefill `prompt + tokens-so-far`) | already running; just continue |
| KV across the sync | dropped on abort; rebuilt on relaunch (radix may rematch) | **retained in place** (`flush_cache=False`) |
| weight update | flushing sync (quiesced) | non-flushing sync on the live scheduler |
| version span | per *sequence*, stitched at carry/merge | per *token*, continuous |
| correctness anchor | recorded rollout logprobs (`old_logp_source=rollout`) | recorded rollout logprobs (`old_logp_source=rollout`) |

Partial rollout is the conservative barrier removal: it still stops generation, but
trades the straggler-drain wait for an abort + re-prefill. In-flight is the
aggressive one: generation literally never stops, at the cost of the hybrid-KV
approximation above.

## Usage

```bash
# async trainer (#68), v2 sglang engine, in-flight weight update ON.
# Most meaningful with max_inflight>1 so a generation is still running when a
# sync fires (with max_inflight=1 + weight_sync_interval=1 + staleness=0 the
# launch clamp consumes each generation before the next sync, so few/no
# sequences actually cross a sync — see below).
ENTRY=train_async_ar bash examples/run_experiment_single_node.sh \
  ar/qwen3_drpo_4b_base_dapo_sglang_async \
  +in_flight_weight_update=true \
  max_inflight=2
```

Requirements / interactions:

- **`algorithm.old_logp_source: rollout`** (the async recipe default) — the whole
  correctness argument rests on the recorded behaviour logprobs. With `replay` the
  per-token version identity is lost and the ratio for cross-version tokens is wrong.
- **`max_inflight > 1`** to realise the benefit. The launch clamp in `_next_batch`
  (the on-policy guarantee for the OFF path) bounds how far ahead generations launch
  by `buffer_max_staleness`. With `max_inflight=1`, `weight_sync_interval=1`,
  `buffer_max_staleness=0` each generation is consumed before the next sync, so
  nothing is in flight at sync time and the feature is a no-op (correct, just
  inert). Raising `max_inflight` (and/or `buffer_max_staleness`) keeps generations
  running across syncs, which is where in-flight update earns its keep. The clamp
  itself is **unchanged** — the on-policy semantics of the OFF path are preserved
  exactly.
- Mutually exclusive with `partial_rollout` (the ctor fails closed if both set).

## Caveats

- **Throughput benefit is regime-dependent**, like partial rollout: it is large when
  the sync-barrier drain dominates per-rollout wall-clock (small batch / short cycle
  relative to generation) and ~zero when the run is generation-throughput-bound
  (large batch — by sync time the in-flight generations are nearly complete anyway).
  In both regimes it should converge identically and never hurt, because the
  recorded-logprob ratio absorbs the version mix.
- **Hybrid KV** (see correctness above) is intrinsic and accepted; it is the
  PipelineRL approximation.
- **No per-token integer version tag** (see correctness above); staleness accounting
  uses the conservative launch-version bound.
- **Not validated on a GPU/cluster in this change** — implemented and statically
  checked only. The mechanism (`flush_cache=False` distributed update on a live
  scheduler + barrier removal + recorded-logprob ratio) is argued above and built on
  the same recorded-logprob property that partial rollout validated end-to-end
  (`ratio_mean≈1.0` across syncs).
