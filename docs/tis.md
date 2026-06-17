# Truncated Importance Sampling (TIS)

Roadmap #94 item #11. Config-gated, **DEFAULT OFF**. Mirrors verl's
`rollout_is` (`actor.policy_loss.rollout_is_*`).

TIS is a *truncated* multiplicative importance-sampling correction for the
distribution mismatch between the **rollout (behaviour) policy μ** that drew the
tokens and the **train policy π_train** that is being optimized. Per token:

```
w_t = min( exp(new_logp_t - rollout_logp_t), C )   # truncate mode
    = min( π_train(a_t|s_t) / μ_rollout(a_t|s_t), C )
```

`C` is `tis_clip` (verl `rollout_is_threshold`, ~2.0). The weight is **detached**
and multiplied onto the per-token policy loss. It is OFF when `tis_clip` is
`None`.

## Config

Added to `GRPOConfig`, `DRPOConfig`, `CPPOConfig` (and the matching `__init__`
kwargs):

| flag | verl name | default | meaning |
| --- | --- | --- | --- |
| `tis_clip` | `rollout_is_threshold` | `null` (OFF) | truncation cap `C`; `~2.0` turns it on |
| `tis_level` | `rollout_is_level` | `token` | `token` \| `sequence` |
| `tis_mode` | `rollout_is_mode` | `truncate` | `truncate` \| `mask` |

YAML example (GRPO):

```yaml
algorithm:
  _target_: unirl.algorithms.grpo.GRPO
  clip_range: 0.2
  tis_clip: 2.0       # ON; omit or null = OFF (default)
  tis_level: token
  tis_mode: truncate
```

### Modes (verl `rollout_is_mode`)

- `truncate` (default): `w = min(IS, C)`. The weight saturates at the cap, but
  the policy surrogate it multiplies **keeps its gradient** — it is a soft
  down-weighting of over-weighted tokens, *not* a hard clip that zeroes the
  gradient the way the PPO `torch.clamp` does. (verl `truncate`, one-sided/upper.)
- `mask`: `w = 1[IS ≤ C]` — a 0/1 keep mask that drops tokens whose train/rollout
  weight exceeds the cap. (verl `mask`.)

### Levels (verl `rollout_is_level`)

- `token` (default): per-token weight.
- `sequence`: one weight per sequence = `exp(Σ_t (new_logp_t − rollout_logp_t))`,
  i.e. the product of the per-token IS weights, truncated once, then broadcast
  back over the sequence's tokens.

## Which log-probs, and why

The weight uses:

- **`new_logp`** = the train-time teacher-forced replay at the *current* weights
  (`stage.replay`, the same π_train the PPO ratio numerator uses).
- **`rollout_logp`** = the rollout engine's emitted log-prob μ. In UniRL this is
  `segment.log_probs` **when `old_logp_source='rollout'`**.

Because the rollout log-prob must be the genuine μ and must stay aligned with the
packed tokens under the train stack's per-micro slicing, TIS **requires
`old_logp_source='rollout'`** (GRPO is always rollout-anchored, so it is always
eligible; DRPO/CPPO raise at construction if `tis_clip` is set with
`old_logp_source='replay'`). In `replay` mode `prepare_segment` *overwrites*
`segment.log_probs` with a train-side anchor, so the original μ is no longer
available — hence the guard. This matches the task's "require
`old_logp_source=rollout` (or the rollout logprob to be available)".

## The key honest question: does TIS add anything beyond UniRL's ratio?

UniRL already folds the rollout policy into the PPO ratio. With
`old_logp_source='rollout'` the PPO ratio denominator **is** the rollout log-prob:

```
r_PPO = exp(new_logp − old_logp) = exp(new_logp − rollout_logp) = π_train / μ_rollout
```

So the PPO ratio *already equals the TIS importance weight*, and PPO already
clips it — but **two-sided** (`clamp(r, 1−ε⁻, 1+ε⁺)`), and that clip **zeroes the
gradient** on the clipped side. This is different from TIS in three ways, all of
which are why TIS can still add a correction here:

1. **One-sided vs two-sided.** TIS truncates only the *upper* tail (`min(·, C)`).
   It targets exploded weights (a token the train policy now finds much more
   likely than the rollout did), which dominate the variance of an IS estimator.
2. **Truncate (keep gradient) vs clip (kill gradient).** PPO's `clamp` makes the
   loss flat → zero gradient for clipped tokens. TIS's `truncate` *rescales* the
   gradient-carrying surrogate by a detached factor; the surrogate still
   contributes a (down-weighted) gradient. This is the "truncate, not hard-clip"
   distinction.
3. **Composition.** TIS is applied **on top of** whatever per-token loss the
   algorithm already produced — *after* PPO's clip (GRPO), *after* DRPO's smooth
   quadratic regularizer, *after* CPPO's Binary-TV keep-mask. It is a separate,
   multiplicative variance-reduction factor, not a replacement for those.

So in the canonical UniRL setting (`old_logp_source='rollout'`), **TIS is a
one-sided, gradient-preserving truncation guard on the same rollout-anchored
ratio the PPO objective already uses** — it does not introduce a *new* policy
ratio, it bounds the influence of the most over-weighted tokens beyond what the
two-sided PPO clip does. Concretely:

- For tokens inside the PPO clip band *and* below `C`: `w ≈ r ≈ 1`, TIS is a
  near-no-op.
- For tokens with `IS > C`: TIS down-weights them (truncate) or drops them
  (mask) while keeping the surrogate differentiable, whereas PPO would have
  hard-clipped (zero-gradient) only if the *signed* ratio·advantage made the
  clipped branch the max.

### The textbook TIS setting (`old_logp_source='replay'`)

In verl's standard recipe `old_logp` is the **train-recomputed** log-prob, so the
PPO ratio is `π_train / π_old_train` (pure on-policy drift) and `rollout_is`
supplies the genuinely *separate* `min(exp(old_logp_train − logp_rollout), C)`
correction for the rollout-vs-train gap. UniRL exposes that setting as
`old_logp_source='replay'`. **TIS is intentionally not wired for that mode here**:
once `prepare_segment` overwrites `segment.log_probs`, the original μ is gone and
would need a new rollout-log-prob field plumbed end-to-end through the rollout
engine and the packed-segment slicer — out of scope for this minimal,
config-gated, default-off change. The guard fails loudly rather than silently
using the train-side anchor as if it were μ.

## Metrics

When on, each AR algorithm emits:

- `tis_weight_mean` — mean truncated weight actually applied.
- `tis_is_ratio_mean` / `tis_is_ratio_max` — the *un*-truncated IS ratio.
- `tis_truncated_fraction` — fraction of tokens (or sequences) with `IS > C`.

## Scope

AR token-level algorithms only (GRPO, DRPO, CPPO). The diffusion algorithms
recompute / self-record their anchor with the same model, so their
rollout-vs-train gap is ~0 by construction and TIS does not apply.
