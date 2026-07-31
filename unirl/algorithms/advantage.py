"""Composable advantage estimators owned by the algorithm layer.

Trainers assemble reward/value inputs and select an estimator. Wire types only
carry the resulting tensors; they do not own GRPO, GDPO, or GAE policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Sequence, runtime_checkable

import torch

from unirl.distributed.tensor.ref import hydrate
from unirl.types.sample import Part, _part_with_field
from unirl.types.sample_id import ancestor_id


@dataclass(frozen=True)
class AdvantageBatch:
    """Estimator inputs independent of trainer and wire-type implementations."""

    rewards: torch.Tensor
    group_ids: Optional[Sequence[str]] = None
    component_rewards: Optional[Mapping[str, torch.Tensor]] = None
    values: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class AdvantageEstimate:
    """Estimator output; value-based methods may additionally return targets."""

    advantages: torch.Tensor
    returns: Optional[torch.Tensor] = None


@runtime_checkable
class AdvantageEstimator(Protocol):
    """Structural interface for reward-to-advantage algorithms."""

    @property
    def requires_group_ids(self) -> bool: ...

    def estimate(self, batch: AdvantageBatch) -> AdvantageEstimate: ...


@dataclass(frozen=True)
class GroupedAdvantageEstimator:
    """GRPO-style grouped or global reward normalization.

    ``exclude_non_finite`` is intended for fault-isolated agent trajectories:
    non-finite rows are excluded from statistics and receive zero advantage.
    ``variance_epsilon`` preserves the historical grouped-Part denominator
    ``sqrt(var + eps)``; agentic normalization uses ``std + eps`` instead.
    """

    scope: str = "group"
    normalize: bool = True
    epsilon: float = 1e-8
    use_global_std: bool = False
    group_std_unbiased: bool = False
    global_std_unbiased: bool = True
    exclude_non_finite: bool = False
    variance_epsilon: bool = True

    def __post_init__(self) -> None:
        if self.scope not in {"group", "global"}:
            raise ValueError(f"GroupedAdvantageEstimator.scope must be 'group' or 'global'; got {self.scope!r}.")
        if self.epsilon < 0:
            raise ValueError(f"GroupedAdvantageEstimator.epsilon must be non-negative; got {self.epsilon}.")
        if self.scope == "global" and self.use_global_std:
            raise ValueError("use_global_std applies to grouped means and is incompatible with scope='global'.")

    @property
    def requires_group_ids(self) -> bool:
        return self.scope == "group"

    def estimate(self, batch: AdvantageBatch) -> AdvantageEstimate:
        rewards = batch.rewards.to(torch.float32)
        if rewards.ndim != 1:
            raise ValueError(f"GroupedAdvantageEstimator expects rewards [N], got shape={tuple(rewards.shape)}.")
        if rewards.numel() == 0:
            return AdvantageEstimate(advantages=rewards.clone())

        valid = torch.isfinite(rewards) if self.exclude_non_finite else torch.ones_like(rewards, dtype=torch.bool)
        if self.scope == "global":
            advantages = self._global(rewards, valid)
        else:
            advantages = self._grouped(rewards, valid, batch.group_ids)
        return AdvantageEstimate(advantages=advantages)

    def _global(self, rewards: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        values = rewards[valid]
        if values.numel() == 0:
            return torch.zeros_like(rewards)
        centered = rewards - values.mean()
        if self.exclude_non_finite:
            centered = torch.where(valid, centered, torch.zeros_like(centered))
        if self.normalize:
            centered = centered / (values.std(unbiased=self.global_std_unbiased) + self.epsilon)
        return torch.where(valid, centered, torch.zeros_like(centered)) if self.exclude_non_finite else centered

    def _grouped(
        self,
        rewards: torch.Tensor,
        valid: torch.Tensor,
        group_ids: Optional[Sequence[str]],
    ) -> torch.Tensor:
        if group_ids is None:
            raise ValueError("GroupedAdvantageEstimator(scope='group') requires group_ids.")
        if len(group_ids) != int(rewards.shape[0]):
            raise ValueError(
                f"GroupedAdvantageEstimator group_ids={len(group_ids)} != rewards={int(rewards.shape[0])}."
            )

        groups = _group_index_map(group_ids)
        advantages = torch.zeros_like(rewards)
        global_values = rewards[valid]
        global_denominator = None
        if self.normalize and self.use_global_std and global_values.numel() > 0:
            global_denominator = global_values.std(unbiased=self.global_std_unbiased) + self.epsilon

        for indices in groups.values():
            index = torch.tensor(indices, dtype=torch.long, device=rewards.device)
            group_rewards = rewards[index]
            group_valid = valid[index]
            finite_rewards = group_rewards[group_valid]
            if finite_rewards.numel() == 0:
                continue

            centered = group_rewards - finite_rewards.mean()
            if self.normalize:
                denominator = global_denominator
                if denominator is None:
                    denominator = self._group_denominator(finite_rewards)
                centered = centered / denominator
            advantages[index] = (
                torch.where(group_valid, centered, torch.zeros_like(centered)) if self.exclude_non_finite else centered
            )
        return advantages

    def _group_denominator(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.variance_epsilon:
            return (rewards.var(unbiased=self.group_std_unbiased) + self.epsilon).sqrt()
        if rewards.numel() == 1:
            return rewards.new_ones(())
        return rewards.std(unbiased=self.group_std_unbiased) + self.epsilon


@dataclass(frozen=True)
class GeneralizedAdvantageEstimator:
    """GAE over one trajectory ``[T]`` or a trajectory batch ``[B, T]``."""

    gamma: float = 1.0
    gae_lambda: float = 0.95

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"GeneralizedAdvantageEstimator.gamma must be in [0, 1]; got {self.gamma}.")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"GeneralizedAdvantageEstimator.gae_lambda must be in [0, 1]; got {self.gae_lambda}.")

    @property
    def requires_group_ids(self) -> bool:
        return False

    def estimate(self, batch: AdvantageBatch) -> AdvantageEstimate:
        rewards = batch.rewards
        values = batch.values
        mask = batch.mask
        if values is None:
            raise ValueError("GeneralizedAdvantageEstimator requires values.")
        if rewards.shape != values.shape:
            raise ValueError(f"GAE rewards shape {tuple(rewards.shape)} != values shape {tuple(values.shape)}.")
        if mask is not None and mask.shape != rewards.shape:
            raise ValueError(f"GAE mask shape {tuple(mask.shape)} != rewards shape {tuple(rewards.shape)}.")
        if rewards.ndim == 1:
            advantages = self._estimate_1d(rewards, values, mask)
        elif rewards.ndim == 2:
            advantages = self._estimate_2d(rewards, values, mask)
        else:
            raise ValueError(f"GAE expects [T] or [B, T], got ndim={rewards.ndim}.")
        return AdvantageEstimate(advantages=advantages, returns=advantages + values)

    def _estimate_1d(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        steps = int(rewards.shape[0])
        next_values = torch.cat([values[1:], values.new_zeros(1)])
        if mask is not None:
            valid = mask.to(dtype=values.dtype)
            next_values = next_values * torch.cat([valid[1:], valid.new_zeros(1)])
        deltas = rewards + self.gamma * next_values - values

        advantages = rewards.new_zeros(steps)
        gae = rewards.new_zeros(())
        for step in range(steps - 1, -1, -1):
            gae = deltas[step] + self.gamma * self.gae_lambda * gae
            if mask is not None:
                gae = gae * mask[step].to(dtype=gae.dtype)
            advantages[step] = gae
        return advantages

    def _estimate_2d(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, steps = rewards.shape
        next_values = torch.cat([values[:, 1:], values.new_zeros(batch_size, 1)], dim=1)
        if mask is not None:
            valid = mask.to(dtype=values.dtype)
            next_values = next_values * torch.cat([valid[:, 1:], valid.new_zeros(batch_size, 1)], dim=1)
        deltas = rewards + self.gamma * next_values - values

        advantages = rewards.new_zeros(batch_size)
        output = rewards.new_zeros(batch_size, steps)
        for step in range(steps - 1, -1, -1):
            advantages = deltas[:, step] + self.gamma * self.gae_lambda * advantages
            if mask is not None:
                advantages = advantages * mask[:, step].to(dtype=deltas.dtype)
            output[:, step] = advantages
        return output


def part_group_ids(part: Part, *, group_layer: Optional[int] = None) -> list[str]:
    """Derive and validate contiguous, uniformly branched lineage groups."""

    sample_ids = part.sample_ids
    if not sample_ids:
        return []
    layer = group_layer if group_layer is not None else max(sample_ids[0].count("/") - 1, 0)
    labels = [ancestor_id(sample_id, layer) for sample_id in sample_ids]
    unique_labels = list(dict.fromkeys(labels))
    if len(labels) % len(unique_labels) != 0:
        raise ValueError(
            f"non-uniform advantage groups (n={len(labels)}, n_groups={len(unique_labels)}); "
            "use Part.fork to build uniformly branched groups."
        )
    branch = len(labels) // len(unique_labels)
    expected = [label for label in unique_labels for _ in range(branch)]
    if labels != expected:
        raise ValueError("advantage groups must be contiguous by parent; got interleaved sample_ids.")
    return labels


def estimate_part_advantages(
    part: Part,
    estimator: AdvantageEstimator,
    *,
    group_layer: Optional[int] = None,
) -> Part:
    """Hydrate one Part, run an estimator, and attach row-aligned advantages."""

    if part.rewards is None:
        raise ValueError("estimate_part_advantages: part has no rewards")
    if not part.sample_ids:
        return part

    rewards = hydrate(part.rewards)
    component_rewards = (
        {name: hydrate(values) for name, values in part.component_rewards.items()}
        if isinstance(part.component_rewards, Mapping)
        else None
    )
    group_ids = part_group_ids(part, group_layer=group_layer) if estimator.requires_group_ids else None
    estimate = estimator.estimate(
        AdvantageBatch(
            rewards=rewards,
            group_ids=group_ids,
            component_rewards=component_rewards,
        )
    )
    if int(estimate.advantages.shape[0]) != len(part.sample_ids):
        raise ValueError(
            f"{type(estimator).__name__} returned {int(estimate.advantages.shape[0])} advantages "
            f"for {len(part.sample_ids)} Part rows."
        )
    return _part_with_field(part, "advantages", estimate.advantages)


def _group_index_map(group_ids: Sequence[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, raw_group_id in enumerate(group_ids):
        if raw_group_id is None:
            raise ValueError(f"advantage group_id at index {index} is None.")
        group_id = str(raw_group_id).strip()
        if not group_id:
            raise ValueError(f"advantage group_id at index {index} is empty.")
        groups.setdefault(group_id, []).append(index)
    return groups


__all__ = [
    "AdvantageBatch",
    "AdvantageEstimate",
    "AdvantageEstimator",
    "GeneralizedAdvantageEstimator",
    "GroupedAdvantageEstimator",
    "estimate_part_advantages",
    "part_group_ids",
]
