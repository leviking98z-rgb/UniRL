"""Canonical post-inference reward operations.

Reward backends own inference and :class:`RewardService` owns attaching scores.
This module owns the remaining framework semantics: materializing actor-returned
reward tensors, computing scalar statistics, and assigning frontier credit back
through a branched Sample lineage. Trainers consume these operations instead of
reimplementing transport and aggregation details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch

from unirl.distributed.tensor.ref import hydrate
from unirl.types.sample import Part, Sample, _part_with_field


@dataclass(frozen=True)
class RewardOutcome:
    """A Sample whose selected Part carries materialized reward fields."""

    sample: Sample
    part_index: int

    @property
    def part(self) -> Part:
        return self.sample.parts[self.part_index]

    @property
    def rewards(self) -> Optional[torch.Tensor]:
        return self.part.rewards

    @property
    def count(self) -> int:
        return int(self.rewards.numel()) if self.rewards is not None else 0

    @property
    def total(self) -> float:
        if self.rewards is None:
            return 0.0
        return float(self.rewards.to(torch.float32).sum().item())

    @property
    def mean(self) -> float:
        if self.rewards is None:
            return 0.0
        return float(self.rewards.to(torch.float32).mean().item())


def materialize_reward(sample: Sample, part_index: int = -1) -> RewardOutcome:
    """Hydrate one Part's reward fields once and return the updated Sample."""

    if not sample.parts:
        raise ValueError("materialize_reward: Sample has no Parts")
    index = part_index if part_index >= 0 else len(sample.parts) + part_index
    if index < 0 or index >= len(sample.parts):
        raise IndexError(f"materialize_reward: part_index {part_index} is out of range for {len(sample.parts)} Parts")

    part = sample.parts[index]
    if part.rewards is not None:
        part = _part_with_field(part, "rewards", hydrate(part.rewards))
    if isinstance(part.component_rewards, dict):
        components = {name: hydrate(value) for name, value in part.component_rewards.items()}
        part = _part_with_field(part, "component_rewards", components)

    parts = list(sample.parts)
    parts[index] = part
    return RewardOutcome(sample=sample.with_parts(parts), part_index=index)


def attach_frontier(reward: Any, sample: Sample) -> Sample:
    """Attach frontier scores without forcing transport materialization.

    Async trainers use this at reap time so reward transfer can remain deferred
    until the buffered sample is selected for training.
    """

    return reward.score_and_attach(sample)


def score_frontier(reward: Any, sample: Sample) -> RewardOutcome:
    """Score and materialize the frontier through the one reward-service API."""

    return materialize_reward(attach_frontier(reward, sample))


def propagate_rewards(
    sample: Sample,
    op: Literal["mean", "max", "sum"] = "mean",
) -> Sample:
    """Assign child rewards to unscored ancestors, from frontier to root.

    Each child Part is ordered in uniform parent groups. Directly scored parents
    win; otherwise the configured reduction collapses each child's branch.
    Reward tensors must already be materialized, normally via
    :func:`score_frontier`.
    """

    new_parts = list(sample.parts)
    for index in range(len(new_parts) - 1, -1, -1):
        part = new_parts[index]
        if part.rewards is not None or index + 1 >= len(new_parts):
            continue
        child = new_parts[index + 1]
        if child.is_root:
            continue
        if child.rewards is None:
            raise ValueError(
                f"propagate_rewards: cannot aggregate from part {index + 1} to {index} — "
                "child.rewards is None. Score the leaf parts first."
            )
        if not torch.is_tensor(child.rewards):
            raise TypeError(
                f"propagate_rewards: child rewards at part {index + 1} are not materialized; "
                "call score_frontier() or materialize_reward() first."
            )

        parent_count = len(part.sample_ids)
        child_count = len(child.sample_ids)
        if parent_count == 0 or child_count % parent_count != 0:
            raise ValueError(
                f"propagate_rewards: non-uniform branching from part {index + 1} ({child_count} samples) "
                f"to {index} ({parent_count} samples). Group-by-parent ordering requires "
                "child_count % parent_count == 0."
            )
        branch = child_count // parent_count
        grouped = child.rewards.view(parent_count, branch)
        if op == "mean":
            aggregated = grouped.mean(dim=1)
        elif op == "max":
            aggregated = grouped.amax(dim=1)
        elif op == "sum":
            aggregated = grouped.sum(dim=1)
        else:
            raise ValueError(f"propagate_rewards: unknown op {op!r}; expected 'mean', 'max', or 'sum'.")
        new_parts[index] = _part_with_field(part, "rewards", aggregated)

    return sample.with_parts(new_parts)


__all__ = [
    "RewardOutcome",
    "attach_frontier",
    "materialize_reward",
    "propagate_rewards",
    "score_frontier",
]
