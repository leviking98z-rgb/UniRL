"""Multi-frame PickScore reward with reward-hacking diagnostics.

Scores every selected frame against the prompt, aggregates those frame scores
into the training reward, and returns per-video diagnostic components:

- first / mean / min / max / last frame PickScore
- frame-score standard deviation and first-minus-mean gap
- pixel motion and first-to-last change
- adjacent-frame CLIP-embedding cosine similarity

The scorer is intended for video-RL diagnosis: ``aggregation="first"``
reproduces the legacy first-frame objective while exposing what the unscored
frames do; ``aggregation="mean"`` trains on the whole video.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

import torch

from unirl.reward.base import BaseRewardComponentSpec
from unirl.types.reward import RewardRequest, RewardResponse
from unirl.utils.media import tensor_frame_to_pil

from .pickscore import PickScoreRewardScorer


def _extract_feature_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0]
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"Unexpected CLIP output format: {type(output)}")


class VideoPickScoreFramesScorer(PickScoreRewardScorer):
    """PickScore every video frame and aggregate per video."""

    canonical_model_name = "videopickscore_frames"
    input_kind = "video"

    def __init__(self, *, config: "VideoPickScoreFramesSpec", base_device: str) -> None:
        self.aggregation = str(config.aggregation).strip().lower()
        if self.aggregation not in {"first", "mean", "min"}:
            raise ValueError(
                f"VideoPickScoreFramesSpec.aggregation must be first|mean|min, got {config.aggregation!r}."
            )
        self.frame_stride = int(config.frame_stride)
        if self.frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {self.frame_stride}.")
        super().__init__(config=config, base_device=base_device)

    @staticmethod
    def _normalize_video(video: torch.Tensor) -> torch.Tensor:
        if not isinstance(video, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor video, got {type(video).__name__}.")
        v = video.detach().cpu()
        if v.dim() == 5:
            v = v.squeeze(0)
        if v.dim() != 4 or int(v.shape[0]) not in (1, 3, 4):
            raise ValueError(f"Expected channel-first [C,T,H,W] video, got {tuple(v.shape)}.")
        if not v.is_floating_point():
            v = v.float() / 255.0
        elif v.numel() > 0 and float(v.max().item()) > 1.0:
            v = v.float() / 255.0
        else:
            v = v.float()
        return v.clamp(0.0, 1.0)

    def _text_features(self, prompts: List[str]) -> torch.Tensor:
        chunks: List[torch.Tensor] = []
        for i in range(0, len(prompts), self.batch_size):
            inputs = self.processor(
                text=prompts[i : i + self.batch_size],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = _extract_feature_tensor(self.model.get_text_features(**inputs))
                emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            chunks.append(emb)
        return torch.cat(chunks, dim=0)

    def _image_features(self, frames) -> torch.Tensor:
        chunks: List[torch.Tensor] = []
        for i in range(0, len(frames), self.batch_size):
            inputs = self.processor(
                images=frames[i : i + self.batch_size],
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = _extract_feature_tensor(self.model.get_image_features(**inputs))
                emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            chunks.append(emb)
        return torch.cat(chunks, dim=0)

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        start = time.time()
        batch_size = request.batch_size
        try:
            videos = request.videos
            prompts = request.prompts
            if videos is None:
                raise ValueError("VideoPickScoreFramesScorer requires request.videos.")
            if len(videos) != len(prompts):
                raise ValueError(f"videos/prompts mismatch: {len(videos)} != {len(prompts)}.")

            normalized: List[torch.Tensor] = []
            frame_pils = []
            frame_owners: List[int] = []
            spans: List[tuple[int, int]] = []
            motion_l1: List[float] = []
            first_last_l1: List[float] = []

            for owner, raw_video in enumerate(videos):
                video = self._normalize_video(raw_video)
                indices = list(range(0, int(video.shape[1]), self.frame_stride))
                if not indices:
                    indices = [0]
                if indices[-1] != int(video.shape[1]) - 1:
                    indices.append(int(video.shape[1]) - 1)
                selected = video[:, indices]
                normalized.append(selected)

                start_idx = len(frame_pils)
                for frame_idx in range(int(selected.shape[1])):
                    frame_pils.append(tensor_frame_to_pil(selected[:, frame_idx]))
                    frame_owners.append(owner)
                spans.append((start_idx, len(frame_pils)))

                if selected.shape[1] > 1:
                    motion_l1.append(float((selected[:, 1:] - selected[:, :-1]).abs().mean().item()))
                    first_last_l1.append(float((selected[:, -1] - selected[:, 0]).abs().mean().item()))
                else:
                    motion_l1.append(0.0)
                    first_last_l1.append(0.0)

            text_embs = self._text_features(prompts)
            image_embs = self._image_features(frame_pils)
            owner_idx = torch.tensor(frame_owners, device=image_embs.device, dtype=torch.long)
            with torch.no_grad():
                logit_scale = self.model.logit_scale.exp()
                frame_scores = logit_scale * (image_embs * text_embs[owner_idx]).sum(dim=-1) / 26.0

            components: Dict[str, List[float]] = {
                "first": [],
                "mean": [],
                "min": [],
                "max": [],
                "last": [],
                "frame_std": [],
                "first_minus_mean": [],
                "mean_minus_min": [],
                "motion_l1": motion_l1,
                "first_last_l1": first_last_l1,
                "temporal_clip_cos": [],
                "temporal_clip_delta": [],
            }
            rewards: List[float] = []

            for start_idx, end_idx in spans:
                scores = frame_scores[start_idx:end_idx]
                embs = image_embs[start_idx:end_idx]
                first = float(scores[0].item())
                mean = float(scores.mean().item())
                minimum = float(scores.min().item())
                maximum = float(scores.max().item())
                last = float(scores[-1].item())
                frame_std = float(scores.std(unbiased=False).item())
                if embs.shape[0] > 1:
                    temporal_cos = float((embs[1:] * embs[:-1]).sum(dim=-1).mean().item())
                else:
                    temporal_cos = 1.0

                components["first"].append(first)
                components["mean"].append(mean)
                components["min"].append(minimum)
                components["max"].append(maximum)
                components["last"].append(last)
                components["frame_std"].append(frame_std)
                components["first_minus_mean"].append(first - mean)
                components["mean_minus_min"].append(mean - minimum)
                components["temporal_clip_cos"].append(temporal_cos)
                components["temporal_clip_delta"].append(1.0 - temporal_cos)

                reward = {"first": first, "mean": mean, "min": minimum}[self.aggregation]
                rewards.append(reward)

            return RewardResponse(
                rewards=rewards,
                component_rewards=components,
                successes=[True] * batch_size,
                errors=[None] * batch_size,
                compute_time=time.time() - start,
            )
        except Exception as exc:
            return RewardResponse(
                rewards=[0.0] * batch_size,
                successes=[False] * batch_size,
                errors=[str(exc)] * batch_size,
                compute_time=time.time() - start,
            )


@dataclass
class VideoPickScoreFramesSpec(BaseRewardComponentSpec):
    """Configuration for multi-frame PickScore diagnostics."""

    batch_size: int = 8
    device: str = "auto"
    processor_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_id: str = "yuvalkirstain/PickScore_v1"
    aggregation: str = "first"
    frame_stride: int = 1


__all__ = ["VideoPickScoreFramesScorer", "VideoPickScoreFramesSpec"]
