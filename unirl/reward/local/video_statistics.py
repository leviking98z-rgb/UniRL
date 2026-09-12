"""Lightweight video-statistics reward for nonzero-gradient smoke tests."""

from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass

import torch

from unirl.reward.base import BaseRewardComponentSpec, RewardBackend
from unirl.types.reward import RewardRequest, RewardResponse

logger = logging.getLogger(__name__)


class VideoStatisticsRewardScorer(RewardBackend):
    """Score sparse video samples without loading an external reward model."""

    input_kind = "video"

    def __init__(self, *, config: "VideoStatisticsRewardSpec", base_device: str = "cpu") -> None:
        del base_device
        super().__init__(model_name="video_statistics_reward", batch_size=1)
        self.sample_frames = max(2, int(config.sample_frames))
        self.spatial_stride = max(1, int(config.spatial_stride))
        self.tie_break_scale = float(config.tie_break_scale)

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        start = time.perf_counter()
        videos = request.generated.get("video")
        if videos is None:
            raise ValueError("VideoStatisticsRewardScorer requires request.generated['video'].")

        rewards = []
        brightnesses = []
        contrasts = []
        motions = []
        media_statistics = []
        tie_breaks = []
        sample_ids = list(request.sample_ids or [])
        group_ids = list(request.group_ids or [])
        for index, video in enumerate(videos.to_list()):
            frames = video.frames
            if frames is None or frames.ndim != 4:
                raise ValueError(
                    "VideoStatisticsRewardScorer expects per-sample frames [T, C, H, W], got "
                    f"{None if frames is None else tuple(frames.shape)}."
                )
            frame_count = min(self.sample_frames, int(frames.shape[0]))
            frame_indices = (
                torch.linspace(0, int(frames.shape[0]) - 1, steps=frame_count, device=frames.device).round().long()
            )
            sampled = frames[frame_indices, :, :: self.spatial_stride, :: self.spatial_stride].detach().float().cpu()
            if not frames.is_floating_point():
                sampled.div_(255.0)
            elif sampled.numel() and float(sampled.max()) > 1.0:
                sampled.div_(255.0)
            sampled.clamp_(0.0, 1.0)

            brightness = float(sampled.mean().item())
            contrast = float(sampled.std(unbiased=False).item())
            motion = float((sampled[1:] - sampled[:-1]).abs().mean().item()) if int(sampled.shape[0]) > 1 else 0.0
            sample_id = sample_ids[index] if index < len(sample_ids) else str(index)
            digest = int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:8], "big") / float(2**64)
            tie_break = self.tie_break_scale * digest
            media_statistic = brightness + 0.25 * contrast + 0.25 * motion
            reward = media_statistic + tie_break

            rewards.append(reward)
            brightnesses.append(brightness)
            contrasts.append(contrast)
            motions.append(motion)
            media_statistics.append(media_statistic)
            tie_breaks.append(tie_break)

        reward_tensor = torch.tensor(rewards, dtype=torch.float64)
        media_group_values: dict[str, list[float]] = defaultdict(list)
        reward_group_values: dict[str, list[float]] = defaultdict(list)
        for index, reward in enumerate(rewards):
            group_id = group_ids[index] if index < len(group_ids) else str(index)
            media_group_values[group_id].append(media_statistics[index])
            reward_group_values[group_id].append(reward)
        media_nonzero_groups = sum(max(values) > min(values) for values in media_group_values.values())
        reward_nonzero_groups = sum(max(values) > min(values) for values in reward_group_values.values())
        logger.info(
            "VIDEO_STATS_REWARD count=%d mean=%.8f std=%.8f min=%.8f max=%.8f "
            "groups=%d media_nonzero_groups=%d reward_nonzero_groups=%d",
            len(rewards),
            float(reward_tensor.mean().item()),
            float(reward_tensor.std(unbiased=False).item()),
            float(reward_tensor.min().item()),
            float(reward_tensor.max().item()),
            len(reward_group_values),
            media_nonzero_groups,
            reward_nonzero_groups,
        )
        return RewardResponse(
            rewards=rewards,
            component_rewards={
                "brightness": brightnesses,
                "contrast": contrasts,
                "motion": motions,
                "media_statistic": media_statistics,
                "identity_tie_break": tie_breaks,
            },
            successes=[True] * len(rewards),
            errors=[None] * len(rewards),
            compute_time=time.perf_counter() - start,
        )

    def is_available(self) -> bool:
        return True


@dataclass
class VideoStatisticsRewardSpec(BaseRewardComponentSpec):
    """Configuration for the sparse video-statistics smoke reward."""

    sample_frames: int = 3
    spatial_stride: int = 32
    tie_break_scale: float = 1.0e-4
    weights: object | None = None
    scorers: object | None = None


__all__ = ["VideoStatisticsRewardScorer", "VideoStatisticsRewardSpec"]
