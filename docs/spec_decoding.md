# Speculative decoding for rollout (experimental, gated, default OFF)

> Roadmap #94, item #9 — **LOW priority. This can be net-negative in RL.** Read
> the staleness hazard below before turning it on.

## The mechanism

Speculative decoding makes a small/cheap **draft** propose several tokens that
the full **policy** then *verifies in one forward pass*: every draft token the
policy would have sampled anyway is accepted for free, so a single verify pass
emits `1 + (accepted)` tokens instead of one. The speedup is entirely a function
of the **acceptance rate** — how often the draft's proposals match the policy.

SGLang implements this server-side (EAGLE / EAGLE3 / NEXTN learned drafts, and a
training-free NGRAM draft). UniRL's v2 SGLang rollout engine already spawns the
SRT server from a config-spelled `ServerArgs` intent
(`SGLangEngineConfig.server_intent` → `HTTPBackend.boot` / `NativeBackend.boot`
filter against the real `ServerArgs` fields and launch). This feature just lets a
recipe spell the `speculative_*` `ServerArgs` knobs through that same path — no
new code path in generation, weight sync, or partial rollout.

## Why a FIXED learned draft decays in RL (the staleness hazard)

In supervised serving the policy is frozen, so a draft trained to mimic it stays
accurate and acceptance stays high. **RL is different: the policy weights change
every optimizer step** (UniRL pushes fresh weights into SGLang every
`weight_sync_interval` rollouts). A draft head trained against weight version `v`
proposes tokens for a policy that has already moved to `v+1, v+2, …`. The
distribution it learned to predict drifts out from under it, acceptance falls,
and each verify pass redeems fewer free tokens — while still paying the draft's
forward cost plus the larger verify batch. Past some point you are doing *more*
work per token than plain decoding.

This is not hypothetical: **verl reports speculative decoding ~50% SLOWER on
H20** in RL rollout for exactly this reason. So a fixed EAGLE/EAGLE3 draft is a
plausible LOSS in RL unless you also keep the draft fresh (see the TODO below).

## The recommended path: a training-free (n-gram / suffix) draft

`speculative_algorithm="NGRAM"` builds its proposals from a **suffix automaton
over the prompt + generated stream itself** — there is no draft model and nothing
to train. Because it reads the *current* output distribution rather than a frozen
snapshot of an old policy, it **self-syncs to whatever the policy is right now**
and survives the per-step weight change far better than a learned draft. It needs
no draft checkpoint, no `speculative_num_steps` / `speculative_eagle_topk`, and
no retraining loop. It wins most on repetitive / structured outputs (math
scaffolding, code, formatted CoT) where suffix matches are frequent.

**This is the recommended option for RL.** It is still gated and still
opportunistic (gain depends on the acceptance rate of suffix matches), but it has
no staleness failure mode.

## The config flag

All knobs live on `SGLangEngineConfig` (`unirl/rollout/engine/sglang/config.py`)
and map 1:1 onto the SGLang `ServerArgs` `speculative_*` fields via
`server_intent`. **`speculative_algorithm` is the gate: `None` (default) = OFF,
and emits NOTHING into the `ServerArgs` intent** — a recipe that does not set it
is byte-for-byte unchanged.

| `SGLangEngineConfig` field | `ServerArgs` field | Notes |
|---|---|---|
| `speculative_algorithm` | `speculative_algorithm` | gate. `None`=off. `NGRAM` / `EAGLE` / `EAGLE3` / `NEXTN` / `STANDALONE`. Case-insensitive. |
| `speculative_draft_model_ckpt_path` | `speculative_draft_model_path` | REQUIRED for the learned drafts (EAGLE/EAGLE3/NEXTN/STANDALONE); unused by NGRAM. |
| `speculative_num_steps` | `speculative_num_steps` | learned-draft depth. |
| `speculative_eagle_topk` | `speculative_eagle_topk` | EAGLE draft branching factor. |
| `speculative_num_draft_tokens` | `speculative_num_draft_tokens` | tokens proposed per verify pass. |

Any further `speculative_*` `ServerArgs` knob (e.g. the NGRAM
`speculative_ngram_*` BFS-breadth / trie-depth tuning, or
`speculative_accept_threshold_*`) rides the existing `engine_kwargs` escape hatch
unchanged — no need to mint a typed field for every one.

`__post_init__` validates: an unknown algorithm fails at config-build time (not
as an opaque SRT launch error), and a learned-draft algorithm with no
`speculative_draft_model_ckpt_path` is rejected. Selecting any learned draft logs
a loud warning pointing here.

### Usage

```bash
# Recommended: training-free n-gram draft (self-syncs, no staleness failure mode)
ENTRY=train_async_ar bash examples/run_experiment_single_node.sh \
  ar/qwen3_drpo_4b_base_dapo_sglang_async \
  +rollout.config.speculative_algorithm=NGRAM \
  +rollout.config.speculative_num_draft_tokens=4

# Learned EAGLE3 draft (NOT recommended for RL — see staleness hazard above)
ENTRY=train_async_ar bash examples/run_experiment_single_node.sh \
  ar/qwen3_drpo_4b_base_dapo_sglang_async \
  +rollout.config.speculative_algorithm=EAGLE3 \
  +rollout.config.speculative_draft_model_ckpt_path=/path/to/eagle3_head \
  +rollout.config.speculative_num_steps=5 \
  +rollout.config.speculative_eagle_topk=8 \
  +rollout.config.speculative_num_draft_tokens=16
```

## Correctness

Speculative decoding is **output-distribution-exact**: SGLang's verify step only
accepts a draft token when it matches what the policy's own sampler would have
produced, and falls back to a normal policy sample otherwise. So the generated
token ids — and therefore the per-token logprobs the engine records
(`return_logprob`) — are the **true behaviour-policy** ids/logprobs, identical in
distribution to what plain decoding would have emitted. Nothing downstream
(reward, advantage, the PPO ratio, partial-rollout carry/merge) sees any
difference; this is a pure rollout-*speed* knob, not a sampling change. The
off-policy gap across weight versions is handled exactly as before, by the
rollout-anchored ratio (`old_logp_source: rollout`).

## Caveats / when NOT to use it

- **Default OFF and experimental.** It is opportunistic at best and a measured
  ~50% loss at worst (verl, learned draft on H20).
- **A fixed learned draft (EAGLE/EAGLE3/NEXTN) goes stale in RL** as the policy
  updates each step → acceptance decays → can be net-negative. If you must use
  one, prefer it only with very frequent draft retraining (see TODO).
- Prefer **NGRAM** (training-free, self-syncing) for RL; it has no staleness
  failure mode but still only helps when suffix matches are frequent.
- The gain depends on acceptance rate, batch size, and KV headroom; at large
  batch the rollout is already throughput-bound and there is little to reclaim.
- **Measure before adopting.** Compare `perf/rollout_time_s` (and verify
  `reward_mean` / `ratio_mean` are unchanged, which they should be by the
  correctness argument) with the flag off vs on for *your* model and data.

## TODO: online draft retraining (the only way a learned draft stays fresh)

The one way to keep a *learned* draft from decaying in RL is to retrain it online
against the current policy — e.g. distill the freshly-synced policy into the
EAGLE head on the same cadence as `weight_sync_interval`, pushing the updated
draft into SGLang alongside the policy weights. UniRL already has the cross-slab
weight-sync machinery the policy uses (`NCCLWeightSync` /
`update_weights_from_distributed`); an online-draft path would reuse it to ship
draft updates too. This is **not implemented** — it is the explicit follow-up hook
for anyone who needs a learned draft to remain viable in RL. Until then, NGRAM is
the staleness-robust default.
