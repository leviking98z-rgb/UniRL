"""Shared frame-selection and aggregation policies for video rewards."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, overload

import torch
from PIL import Image

from unirl.types.primitives import Video
from unirl.utils.media import tensor_frame_to_pil

FrameIndexRounding = Literal["floor", "nearest"]


def video_frame_to_pil(frame: torch.Tensor) -> Image.Image:
    """Convert one canonical ``[C, H, W]`` video frame to PIL."""
    if not isinstance(frame, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor frame, got {type(frame).__name__}")
    if frame.ndim != 3:
        raise ValueError(f"Expected a [C, H, W] frame, got shape {tuple(frame.shape)}")

    frame = frame.detach().cpu()
    if not frame.is_floating_point():
        frame = frame.float() / 255.0
    elif frame.numel() > 0 and frame.max() > 1.0:
        frame = frame / 255.0
    return tensor_frame_to_pil(frame.clamp(0.0, 1.0))


def first_video_frame_to_pil(video: Video) -> Image.Image:
    """Select a video's first frame and convert it to PIL."""
    return video_frame_to_pil(video.first_frame())


def sample_video_frames(
    video: Video,
    count: int,
    *,
    rounding: FrameIndexRounding = "floor",
) -> Video:
    """Select exactly ``count`` uniformly-spaced frames.

    ``rounding`` is explicit because existing reward estimators use both
    floor and nearest-index sampling. Keeping the policy at the call site
    avoids changing their numerical behavior during deduplication.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError(f"count must be a positive integer, got {count!r}")
    if rounding not in {"floor", "nearest"}:
        raise ValueError(f"rounding must be 'floor' or 'nearest', got {rounding!r}")

    return Video(frames=video.sample_uniform(count, rounding=rounding))


def sample_video_frames_to_pils(
    video: Video,
    count: int,
    *,
    rounding: FrameIndexRounding = "floor",
) -> list[Image.Image]:
    """Select uniform frames and convert them to PIL images."""
    sampled = sample_video_frames(video, count, rounding=rounding)
    return [video_frame_to_pil(frame) for frame in sampled.as_tchw()]


@overload
def mean_frame_scores(scores: torch.Tensor, frames_per_video: int) -> torch.Tensor: ...


@overload
def mean_frame_scores(scores: Sequence[float], frames_per_video: int) -> list[float]: ...


def mean_frame_scores(
    scores: torch.Tensor | Sequence[float],
    frames_per_video: int,
) -> torch.Tensor | list[float]:
    """Mean-reduce flat frame scores into one score per video."""
    if not isinstance(frames_per_video, int) or isinstance(frames_per_video, bool) or frames_per_video <= 0:
        raise ValueError(f"frames_per_video must be a positive integer, got {frames_per_video!r}")

    if isinstance(scores, torch.Tensor):
        if scores.ndim != 1:
            raise ValueError(f"Expected flat frame scores, got shape {tuple(scores.shape)}")
        if scores.numel() % frames_per_video:
            raise ValueError(
                f"Frame score count {scores.numel()} is not divisible by frames_per_video={frames_per_video}"
            )
        return scores.reshape(-1, frames_per_video).mean(dim=1)

    values = list(scores)
    if len(values) % frames_per_video:
        raise ValueError(f"Frame score count {len(values)} is not divisible by frames_per_video={frames_per_video}")
    return [
        sum(values[start : start + frames_per_video]) / frames_per_video
        for start in range(0, len(values), frames_per_video)
    ]


def temporal_consistency_score(video: Video) -> float:
    """Return ``1 - mean(abs(frame[t+1] - frame[t]))``, clamped at zero."""
    frames = video.as_tchw()
    frame_diffs = []
    for index in range(len(frames) - 1):
        frame_diffs.append((frames[index] - frames[index + 1]).abs().mean().item())
    average_diff = sum(frame_diffs) / len(frame_diffs) if frame_diffs else 0.0
    return max(0.0, 1.0 - average_diff)


__all__ = [
    "FrameIndexRounding",
    "first_video_frame_to_pil",
    "mean_frame_scores",
    "sample_video_frames",
    "sample_video_frames_to_pils",
    "temporal_consistency_score",
    "video_frame_to_pil",
]
