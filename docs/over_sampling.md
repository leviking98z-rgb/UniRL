# Over-sampling + dynamic sampling (DAPO)

Roadmap issue #94, item #3. Two cooperating knobs on `AsyncARTrainer` that kill
the **across-group straggler** and filter **no-signal** groups. Both default OFF —
existing recipes and the exact-fit async path are byte-for-byte unchanged.

## The problem

The async trainer already hides the *within-group* straggler: generations are
non-blocking Ray futures, `max_inflight` overlaps them, and a staleness-bounded
group buffer lets fast generations bank while slow ones run. But a rollout still
needs `batch_size` **complete groups** before it can train, so one slow group
(e.g. a prompt whose samples all run to `max_new_tokens`) gates the whole step —
the *across-group* straggler.

Separately, GRPO gives **zero gradient** to a group whose samples all receive the
identical reward: the group advantage is `(r - group_mean) / group_std`, and a
uniform-reward group has `group_std = 0`, so every sample's advantage is `0`.
All-correct and all-wrong prompts are pure overhead — they consume rollout
compute and training FLOPs but move no weights.

## Mechanism

A rollout's worth of data is one GRPO group per prompt — a `RolloutTrack` of
`samples_per_prompt` scored samples (`track.split()`), stamped with a
`weight_version` and a monotonic `gen_id`, held in the `_RolloutBuffer`.

### Over-sampling (`over_sampling_batch_size`)

Gather `target = max(batch_size, over_sampling_batch_size)` **valid** groups
before consuming. `_next_batch_oversampled` then drains the **freshest**
`batch_size` (`_RolloutBuffer.drain_freshest`) and **carries the surplus
forward** in the buffer for the next consume. The across-group straggler is no
longer on the critical path: the rollout trains on whichever `target` groups
finished first and defers the slow ones.

Launches are **demand-driven**: while short of `target` and under `max_inflight`,
launch more generations (one generation = up to `batch_size` groups). Needs
`max_inflight > 1` to actually overlap the extra generations — at
`max_inflight = 1` they run serially and over-sampling only adds latency.

### Dynamic sampling (`dynamic_sampling`, DAPO)

At the single buffer-entry chokepoint (`_score_into_buffer`), drop any group whose
samples all got the same reward (`_group_has_no_signal`: `max == min`, hydrating
the worker `TensorRef` first; a single-sample group is always no-signal). Filtered
groups never enter the buffer, so the loop keeps sampling until `target` **valid**
groups are collected — the DAPO "keep the batch full of learnable groups" recipe.

The two compose: `over_sampling_batch_size: 96, dynamic_sampling: true` over-samples
to 96 valid (non-zero-advantage) groups, trains on the freshest 64, recycles 32.

## Recycle vs discard — the choice

Surplus groups are **recycled**, not discarded. `drain_freshest(batch_size)` already
pops the freshest `batch_size` and leaves the rest in the buffer; the next consume
sees them again (subject to staleness eviction). Rationale:

- **No wasted compute.** A completed, scored, valid group is expensive (a full
  generation + reward). Discarding it throws away rollout FLOPs for nothing.
- **Zero new machinery.** The buffer's carry-forward is exactly the existing
  off-policy continuous-buffer behavior; over-sampling just raises the fill
  threshold. Discard would need a separate eviction path and lose data.
- **Staleness already bounds it.** A recycled group cannot live forever: when
  `buffer_max_staleness` is set, `count_fresh` / `drain_freshest` evict groups
  older than `current_version - max_staleness` weight versions, so recycled
  surplus is consumed within the staleness window or dropped — never stale.

The slow/surplus *in-flight* generations are not aborted (that is partial
rollout's job, below); only the across-group **wait** is removed.

## Configuration

```yaml
# examples/ar/qwen3_grpo_4b_base_dapo_sglang_async.yaml (or any async recipe)
max_inflight: 4                 # > 1 so the extra generations overlap
over_sampling_batch_size: 96    # gather 96 valid groups, train freshest 64, recycle 32
dynamic_sampling: true          # drop all-same-reward (zero-advantage) groups
```

CLI override:

```bash
ENTRY=train_async_ar bash examples/run_experiment_single_node.sh \
  ar/qwen3_grpo_4b_base_dapo_sglang_async \
  max_inflight=4 over_sampling_batch_size=96 dynamic_sampling=true
```

Constraints: `over_sampling_batch_size` is `0` (off) or `>= batch_size` (it is the
OVER count) — enforced in the ctor. `dynamic_sampling` works standalone (no
over-sampling): it just keeps the buffer topped with `batch_size` valid groups.

## Correctness

- **GRPO group contiguity.** The buffer's unit is a whole group from
  `track.split()` — siblings stay consecutive within a group, and groups are
  concatenated (`RolloutTrack.concat`) before `compute_advantages`, which still
  sees uniform, group-by-parent-contiguous data. Filtering and recycling operate
  on whole groups; a group is never split, so the `compute_advantages` reshape
  invariant (`n % n_groups == 0`, contiguous `parent_ids`) is preserved.
- **Staleness.** `drain_freshest` and `count_fresh` apply the same eviction
  (`current_version - weight_version <= max_staleness`), so the off-policy bound
  is identical to the exact-fit path. With `stale = 0` (on-policy), all
  generations launched for a consume are under the current `weight_version` and
  are drained at the upcoming sync, so `ratio ≈ 1` still holds — over-sampling
  only changes *how many* generations run per consume, not *which version* they
  are.
- **No deadlock.** When short of `target` with nothing in flight, the
  demand-driven gate always has launch budget (up to `max_inflight`) and launches
  at least one generation — so the loop always makes progress. If
  `dynamic_sampling` filters everything (pathological all-uniform-reward data),
  the trainer keeps sampling (DAPO's intended behavior) and logs a periodic
  heavy-filtering warning rather than hanging.

## Interaction with partial rollout (`partial_rollout`)

Orthogonal and composable. Partial rollout interrupts in-flight generations at the
weight-sync boundary and carries the unfinished ones for continuation; it never
splits a group (whole generations are carried). The DAPO filter still applies at
the single chokepoint: a carried generation that *completes* routes through
`_ingest_partial → _score_into_buffer`, where no-signal groups are dropped just as
on the one-shot path. Carried (incomplete) generations are not yet in the buffer,
so they are counted only as in-flight by the over-sampling launch gate. A
continuation generation may hold fewer than `batch_size` samples (only the
unfinished subset), so the gate's optimistic `inflight * batch_size` group estimate
can slightly over-count under partial rollout — this affects launch *concurrency*
only, never correctness (the loop still always reaches `target`).

## Resume caveat

The exact-fit async path resumes deterministically because launches are 1:1 with
`rollout_id` (replay `start_rollout` `get_samples` calls). With over-sampling /
dynamic-sampling, a rollout consumes a variable number of generations, so the 1:1
identity no longer holds and checkpoint resume fast-forwards by the recorded
rollout count only approximately — consistent with the existing seed-dependent
resume caveat in the synchronous AR trainer. Fresh runs are unaffected; the
default-off path keeps its exact resume.

## Tested

GPU-free static validation: every changed module compiles
(`python -m py_compile`); `_group_has_no_signal`, the buffer `count_fresh` /
`drain_freshest` recycle semantics, and the demand-gate no-deadlock argument
reasoned through above. **NOT** run on a cluster (no GPU available) — no
end-to-end reward / throughput numbers yet.
