"""DigenRL Generation-Axis Pipeline (GAP) — finest-grain micro-batching.

DigenRL's GAP forms pipeline units along the *generation/sampling* axis rather
than the prompt/batch axis: under the small batches typical of diffusion RL
(dozens of samples), splitting by prompt yields only a couple of pipeline units,
so generator and trainer barely overlap. GAP instead makes each *sample* its own
micro-batch, maximizing the number of pipeline units M so a disaggregated
generator/trainer can overlap finely (bubble ~ (gen+train)/M shrinks with M).

This is the train-side realization: a planner that emits **single-sample micros**
(M = batch_size) regardless of the recipe ``micro_batch_size``. The per-rank micro
count stays uniform across DP ranks (count-based geometry), so no NCCL parity
collective is needed. Math is unchanged vs CountPlanner — gradients accumulate
over the same samples, only the accumulation granularity (and thus pipeline
overlap potential) differs (order-invariant, see DigenRL GAP §3.2.1).
"""

from __future__ import annotations

from unirl.algorithms import StageAlgorithm
from unirl.train.stack.planner.count import _count_plan
from unirl.train.stack.planner.types import Plan
from unirl.types.rollout_resp import RolloutTrack


class GAPPlanner:
    """Generation-axis pipeline planner: one sample per micro (maximizes M)."""

    def arrange(
        self, resp_track: RolloutTrack, *, num_updates: int, micro_batch_size: int
    ) -> tuple[RolloutTrack, Plan]:
        # Force single-sample micros (finest pipeline granularity); ignore the
        # recipe micro_batch_size on purpose — that is the point of GAP.
        return resp_track, _count_plan(
            total=int(resp_track.batch_size),
            num_updates=num_updates,
            micro_batch_size=1,
        )

    def validate(self, algorithm: StageAlgorithm) -> None:
        return None
