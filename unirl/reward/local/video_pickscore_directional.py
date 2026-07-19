"""VideoPickScoreDirectional — prompt-directional PickScore with headroom.

Plain PickScore on a T2V output saturates near its ceiling (~0.68) because a
generically-good frame already scores high regardless of the prompt, leaving
little training headroom (flat reward curve). This scorer subtracts the frame's
PickScore against a generic BASELINE caption, so the reward measures how much
better the frame matches THIS prompt than a generic one:

    reward = pickscore(frame, prompt) - lambda_base * pickscore(frame, baseline_caption)

A frame that's generically nice but off-prompt nets ~0; the reward only climbs as
the model actually makes the frame specific to its prompt. This is the T2V analogue
of PR#113's edit-delta (which subtracts source-similarity for V2V) — it gives the
reward headroom from ~0 without needing a source video.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from unirl.reward.base import BaseRewardComponentSpec
from unirl.types.reward import RewardRequest

from .video_pickscore import VideoPickScoreScorer


class VideoPickScoreDirectionalScorer(VideoPickScoreScorer):
    canonical_model_name = "videopickscore_directional"
    input_kind = "video"

    def __init__(self, *, config: "VideoPickScoreDirectionalSpec", base_device: str) -> None:
        self.lambda_base = float(getattr(config, "lambda_base", 1.0))
        self.baseline_caption = str(getattr(config, "baseline_caption", "a video"))
        super().__init__(config=config, base_device=base_device)

    def _score(self, images, texts):
        """PickScore (logit_scale·cos/26) for paired images/texts lists."""
        out: List[float] = []
        for i in range(0, len(images), self.batch_size):
            bi = images[i:i + self.batch_size]; bt = texts[i:i + self.batch_size]
            ii = self.processor(images=bi, padding=True, truncation=True, max_length=77, return_tensors="pt")
            ii = {k: v.to(self.device) for k, v in ii.items()}
            ti = self.processor(text=bt, padding=True, truncation=True, max_length=77, return_tensors="pt")
            ti = {k: v.to(self.device) for k, v in ti.items()}
            with torch.no_grad():
                ie = self.model.get_image_features(**ii)
                ie = (ie if torch.is_tensor(ie) else ie.pooler_output)
                ie = ie / ie.norm(p=2, dim=-1, keepdim=True)
                te = self.model.get_text_features(**ti)
                te = (te if torch.is_tensor(te) else te.pooler_output)
                te = te / te.norm(p=2, dim=-1, keepdim=True)
                sc = (self.model.logit_scale.exp() * (te @ ie.T)).diag() / 26.0
            out.extend(sc.cpu().tolist())
        return out

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        if not request.is_video:
            return super()._compute_model_rewards(request)
        frames = [self._extract_first_frame(v) for v in request.videos]
        prompts = request.prompts
        n = len(frames)
        if len(prompts) != n:
            raise ValueError(f"directional: frames={n} prompts={len(prompts)} mismatch")
        s_prompt = self._score(frames, prompts)
        s_base = self._score(frames, [self.baseline_caption] * n)
        return [p - self.lambda_base * b for p, b in zip(s_prompt, s_base)]


@dataclass
class VideoPickScoreDirectionalSpec(BaseRewardComponentSpec):
    batch_size: int = 8
    device: str = "auto"
    processor_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_id: str = "yuvalkirstain/PickScore_v1"
    lambda_base: float = 1.0
    baseline_caption: str = "a video"
