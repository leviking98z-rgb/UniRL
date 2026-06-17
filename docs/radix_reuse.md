# Prefix/radix cache reuse across resume

Roadmap issue #94, item #8. Companion to [`partial_rollout.md`](partial_rollout.md).

## The problem

Partial rollout resumes a paused sequence by re-submitting `prompt +
tokens-generated-so-far` (see `_partial_rollout.py` / `partial_budget`). If the
SGLang **RadixAttention** prefix cache still holds that prefix, the resume is a
cheap *append*: the engine re-prefills only the one new token and reuses the
already-computed KV for the prefix. A controlled probe (recorded in
`partial_rollout.md`) shows exactly this when the cache survives:

```
resume round 1: prefix_in=351  cached-token=350  new-token=1
resume round 2: prefix_in=607  cached-token=606  new-token=1
```

But today the cache does **not** survive between rounds. Two places wipe it:

1. `SGLangRolloutEngine.sleep()` flushes the cache before releasing memory
   (the colocate path offloads the engine every step).
2. Every weight sync flushes (`update_weights_from_distributed(..., flush_cache=True)`
   on the last bucket).

With `weight_sync_interval=1` the cache is wiped on *every* step, so every
engine-path resume re-prefills the whole grown prefix from scratch — a tax that
grows linearly with the sequence length (observed `cached-token=0` across all
engine-path resumes). The longer the rollout, the worse it gets.

## The correctness constraint (why "just keep the cache" is WRONG)

Cached KV is **computed under a specific weight version**. RadixAttention keys a
prefix to its token ids, *not* to the weights that produced its KV. After a
weight update the cached KV for any prefix is **stale**: reusing it would mix KV
from the old policy with new-policy decode, silently corrupting both the
generated tokens and — fatally for RL — the recorded behaviour-policy logprobs
(`old_logp_source=rollout` reads exactly these).

So radix reuse is valid **only within a single weight version**. The naive fix
("stop flushing, keep the cache") is incorrect: it would serve stale KV the
moment weights change. The cache MUST be flushed whenever weights actually
change, and may be retained only across offloads/resumes that do **not** cross a
weight update.

## The design: conditional, weight-version-bound flush

A new config flag `rollout.config.radix_reuse` (`SGLangEngineConfig`, default
`false`) turns the unconditional flush into a **conditional** one, gated on
whether the radix cache is stale. The engine becomes the single authority on
radix-cache validity within a weight version, via one bit of state:

- `_radix_dirty` — set `True` whenever base weights are received
  (`update_weights_from_distributed` / `update_weights_from_tensor`): the cached
  KV is now from a prior weight version. It is cleared (`False`) by any flush.
  (Per bucket the flag is `not flush_cache`, so a multi-bucket sync that flushes
  on its last bucket ends up clean; a sync configured with `flush_cache=false`
  leaves it dirty until the guard below flushes.)

Three flush sites now consult the flag (all no-ops / unchanged when
`radix_reuse` is off):

- **`sleep()`** — when `radix_reuse` is off it always flushes before releasing
  the KV pool (today's behaviour, byte-for-byte). When on, it flushes **only if
  `_radix_dirty`**, so an offload within the same weight version retains the
  prefix cache for the next resume.
- **weight sync** — unchanged: it flushes on its last bucket exactly when
  weights change (`flush_cache=true` in the `sync:` block, the default). This is
  the weight-version boundary where stale KV must be wiped.
- **`generate()` guard** — a belt-and-braces check: if `radix_reuse` is on and
  `_radix_dirty` is still set when a generation starts (e.g. a sync was
  configured with `flush_cache=false`, or no offload flushed in between), the
  engine flushes first. This guarantees stale KV is **never** served, regardless
  of how the offload/sync flush was configured. The result is exactly **one
  flush per weight version** and never zero.

Net effect: the radix cache is flushed **iff weights actually changed** since
the last flush, and is otherwise retained — the resume becomes the cheap append
the prefix cache is designed for.

### Why correctness is preserved

The only way cached KV from a prior weight version can reach a `generate` is if
`_radix_dirty` were `True` at that `generate`. But `_radix_dirty` is set on every
weight update and cleared only by an actual flush, and `generate()` flushes
first whenever it is still set. Therefore any prefix served from the cache was
computed under the current weight version. Reuse never crosses a weight update.

## The flag

```yaml
rollout:
  config:
    _target_: unirl.rollout.engine.sglang.config.SGLangEngineConfig
    radix_reuse: true        # default false = today's always-flush behaviour
    partial_budget: 256      # the resume mechanism this accelerates
sync:
  flush_cache: true          # keep true; the weight-version flush boundary
```

Or from the CLI on an existing recipe (default-off means recipes are
byte-for-byte unchanged when the flag is unset):

```bash
ENTRY=train_async_ar bash examples/run_experiment_single_node.sh \
  ar/qwen3_drpo_4b_base_dapo_sglang_async \
  +rollout.config.partial_budget=256 \
  +rollout.config.radix_reuse=true \
  weight_sync_interval=2
```

## The `weight_sync_interval` (wsi) interaction

Reuse only exists *within* a weight version, so the win scales with how many
resume rounds fall between two weight syncs — i.e. with `weight_sync_interval`:

| `wsi` | Weights change… | Radix reuse |
|---|---|---|
| `1` | every step | **none** — every step is a weight change, so the cache is (correctly) flushed every step. No prefix survives to be reused. |
| `> 1` | every `wsi` steps | resumes that fall inside a sync window (do not cross a sync) reuse the prefix; only the resume immediately after a sync re-prefills (weights changed → correct flush). |

The benefit therefore grows with `wsi`: the more steps a sequence's resume
rounds spend under one weight version, the more re-prefill work the retained
cache saves. At `wsi=1` `radix_reuse` is a safe no-op — the cache must be (and
is) flushed every step, so enabling the flag changes nothing.

### wsi=1 PD-resume note

Prefill/decode-style resume (partial rollout's re-submit) at `weight_sync_interval=1`
gets **no** radix reuse, and this is correct, not a regression: with weights
changing every step, a sequence that is carried across the sync boundary
(`_relaunch_carry`) is resumed under *new* weights, so its prior-version KV is
stale and must be discarded. The carry still works — the recorded
behaviour-policy logprobs make the off-policy continuation valid
(`old_logp_source=rollout`) — but the resume pays a full re-prefill of the grown
prefix, exactly as before this feature. To actually realise radix reuse for
partial-rollout resumes, run `wsi > 1` so consecutive resume rounds share a
weight version. The reward/ratio behaviour is unchanged either way (this is a
pure prefill-cost optimisation; it never alters which KV the model decodes
from).

## Caveats

- **Resident engine is where this pays off.** The async/disaggregated trainer
  keeps the engine resident (it never calls `sleep()`), so between weight syncs
  the cache naturally survives and `radix_reuse` lets `wsi>1` windows reuse
  prefixes. The colocate trainer offloads the engine every step via a *full*
  `sleep()` (releases the KV pool to hand the GPU to the FSDP shard); the KV
  memory is freed regardless of the flush, so colocate cannot retain the radix
  cache across a step without a weights-only offload (out of scope here, and at
  odds with colocate's memory-sharing model). On the colocate path the flag is a
  no-op-shaped safety change: it suppresses the redundant *flush*, but the KV
  pool is still released.
- **Keep `sync.flush_cache: true`.** It is the natural weight-version flush
  boundary. Setting it to `false` is tolerated (the `generate()` guard still
  flushes a dirty cache before use, so correctness holds) but pointless — the
  flush just moves to the next `generate`.
- **KV headroom.** Retaining the radix cache keeps more of the KV pool occupied
  between resumes; under tight `mem_fraction_static` the pool can still evict the
  prefix under concurrency (LRU), in which case the resume re-prefills anyway —
  reuse is opportunistic, never guaranteed.
- **No reward/ratio impact.** This changes only *when* the prefix KV is
  recomputed, never *what* the model decodes or which logprobs are recorded, so
  it is reward- and importance-ratio-neutral by construction.

## Validation

Static only (no GPU/cluster available for this change): `py_compile` of the
changed modules passes; the default path (`radix_reuse: false`) is byte-for-byte
the prior behaviour (`sleep()` always flushes; weight sync flushes on its last
bucket; the new `_radix_dirty` bookkeeping is read only under the
`radix_reuse` guard). End-to-end reward/throughput numbers are left for a GPU
run; the mechanism is the same prefix-cache reuse already probed in
`partial_rollout.md` ("the resume is nearly free when the radix/prefix cache
survives").
