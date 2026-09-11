"""Return deterministic video rewards for systems benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

from unirl.reward.base import BaseRewardComponentSpec, RewardBackend
from unirl.types.reward import RewardRequest, RewardResponse


class ConstantVideoRewardScorer(RewardBackend):
    """Return one configured score per generated video."""

    input_kind = "video"

    def __init__(self, *, config: "ConstantVideoRewardSpec", base_device: str = "cpu") -> None:
        del base_device
        super().__init__(model_name="constant_video_reward", batch_size=1)
        self.value = float(config.value)

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        batch_size = request.batch_size
        return RewardResponse(
            rewards=[self.value] * batch_size,
            successes=[True] * batch_size,
            errors=[None] * batch_size,
            compute_time=0.0,
        )

    def is_available(self) -> bool:
        return True


@dataclass
class ConstantVideoRewardSpec(BaseRewardComponentSpec):
    value: float = 0.0
    # The systems benchmark overrides a production composite reward in-place.
    # Hydra merges mappings recursively, so retain nullable compatibility
    # fields after the benchmark config clears the inherited mappings.
    weights: object | None = None
    scorers: object | None = None


__all__ = ["ConstantVideoRewardScorer", "ConstantVideoRewardSpec"]
