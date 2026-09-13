"""Fixed-count micro-batching — the default planner."""

from __future__ import annotations

from unirl.algorithms.base import StageAlgorithm
from unirl.train.stack.planner.types import Plan, _build_micro_batch_slices, _update_ranges
from unirl.types.sample import Part
from unirl.types.sample_id import branch_of


def _count_plan(*, total: int, num_updates: int, micro_batch_size: int) -> Plan:
    """Fixed-count plan: contiguous equal updates, each split into ``micro_batch_size`` micros."""
    plan: Plan = []
    for u_start, u_end in _update_ranges(total_size=total, num_updates=num_updates):
        plan.append(
            [
                (u_start + ms, u_start + me)
                for ms, me in _build_micro_batch_slices(total_size=u_end - u_start, micro_batch_size=micro_batch_size)
            ]
        )
    return plan


class CountPlanner:
    """Fixed-count micro-batches: every micro holds ``micro_batch_size`` samples."""

    def arrange(self, part: Part, *, num_updates: int, micro_batch_size: int) -> tuple[Part, Plan]:
        return part, _count_plan(
            total=int(part.batch_size),
            num_updates=num_updates,
            micro_batch_size=micro_batch_size,
        )

    def validate(self, algorithm: StageAlgorithm) -> None:
        return None


def _group_interleaved_permutation(part: Part, *, num_updates: int) -> list[int]:
    """Order each update by the same sibling slice from every contiguous prompt group."""
    total = int(part.batch_size)
    updates = len(_update_ranges(total_size=total, num_updates=num_updates))
    sample_ids = part.validated_sample_ids(context="GroupInterleavedCountPlanner")
    group_ids = part.group_ids
    groups: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    start = 0
    while start < total:
        group_id = group_ids[start]
        end = start + 1
        while end < total and group_ids[end] == group_id:
            end += 1
        if group_id in seen:
            raise ValueError(
                "GroupInterleavedCountPlanner requires each prompt group to be contiguous; "
                f"group {group_id!r} appears in multiple ranges."
            )
        seen.add(group_id)
        groups.append((group_id, start, end))
        start = end

    group_sizes = {end - start for _, start, end in groups}
    if len(group_sizes) != 1:
        raise ValueError(
            f"GroupInterleavedCountPlanner requires uniform prompt-group sizes; observed sizes {sorted(group_sizes)}."
        )
    group_size = group_sizes.pop()
    configured_group_size = getattr(part.sampling_params, "samples_per_prompt", None)
    if configured_group_size is not None and group_size != int(configured_group_size):
        raise ValueError(
            "GroupInterleavedCountPlanner requires complete prompt groups matching "
            f"sampling_params.samples_per_prompt ({int(configured_group_size)}); observed size {group_size}."
        )
    if group_size % updates:
        raise ValueError(
            f"GroupInterleavedCountPlanner requires group_size ({group_size}) to be divisible by "
            f"num_updates_per_batch ({updates})."
        )

    for group_id, group_start, group_end in groups:
        ordinals = [branch_of(sample_id) for sample_id in sample_ids[group_start:group_end]]
        expected = list(range(group_size))
        if ordinals != expected:
            raise ValueError(
                "GroupInterleavedCountPlanner requires sibling branch ordinals 0..group_size-1 "
                f"in every prompt group; group {group_id!r} has {ordinals}, expected {expected}."
            )

    siblings_per_update = group_size // updates
    return [
        group_start + sibling
        for update in range(updates)
        for _, group_start, _ in groups
        for sibling in range(update * siblings_per_update, (update + 1) * siblings_per_update)
    ]


class GroupInterleavedCountPlanner:
    """Opt-in fixed-count updates that preserve sibling membership across DP/SP topology changes."""

    def arrange(self, part: Part, *, num_updates: int, micro_batch_size: int) -> tuple[Part, Plan]:
        permutation = _group_interleaved_permutation(part, num_updates=num_updates)
        return part.select(permutation), _count_plan(
            total=int(part.batch_size),
            num_updates=num_updates,
            micro_batch_size=micro_batch_size,
        )

    def validate(self, algorithm: StageAlgorithm) -> None:
        return None
