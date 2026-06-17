"""Stage-driven ``GRPO`` over a ``TextSegment``.

Implements :class:`StageAlgorithm` and shares the module-level
``_grpo_clip_loss`` / ``_resolve_clip_range_from_schedule`` helpers (in
:mod:`unirl.algorithms.base`) with :class:`FlowGRPO` so their loss
math stays identical. The teacher-forced forward and per-token log-prob
recompute are owned by ``stage.replay(...)``; the algorithm is ~20 lines of
ratio-clip math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Type

import torch

from unirl.types.conditions import Condition
from unirl.types.segments.text import TextSegment

from .base import (
    AlgorithmStepResult,
    BaseAlgorithmConfig,
    StageAlgorithm,
    _grpo_clip_loss,
    _resolve_clip_range_from_schedule,
    _validate_tis_level,
    _validate_tis_mode,
    rollout_replay_logp_absdiff,
    tis_correction,
    typed_conditions,
)


@dataclass
class GRPOConfig(BaseAlgorithmConfig):
    stage_attr: str = "ar"
    conditions_cls: str = ""
    clip_range: float = 1e-4
    clip_schedule: str = "constant"
    # Truncated Importance Sampling (TIS, verl rollout_is). DEFAULT OFF (None).
    # ~2.0 turns it on; see docs/tis.md. tis_level: token|sequence;
    # tis_mode: truncate|mask. Mirrors verl rollout_is_threshold/level/mode.
    tis_clip: Optional[float] = None
    tis_level: str = "token"
    tis_mode: str = "truncate"


class GRPO(StageAlgorithm):
    """GRPO over an AR ``TextSegment`` via ``ARStage.replay``.

    The teacher-forced forward and per-token log-prob recompute is owned by
    :meth:`ARStage.replay`; this class expands per-sample advantages to per-
    token via ``cu_seqlens`` and runs the same PPO clip math.

    Args:
        stage: The :class:`ARStage` whose ``replay`` produces packed-varlen
            new log-probs aligned with ``segment.log_probs``.
        clip_range: PPO clip range epsilon.
        clip_schedule: ``"constant"``, ``"linear_decay"``, or
            ``"cosine_decay"``.
        conditions_cls: Stage-typed conditions container with
            ``from_dict(Mapping[str, Condition])``.
        sampling_temperature: AR rollout temperature, applied as a
            ``logits / T`` scaling inside :meth:`ARStage.replay` so
            replay's log-softmax matches SGLang's sampling distribution
            (``log_softmax(logits / T)``). Injected at construction time
            from the rollout engine config; falls back to
            :class:`ARSamplingParams` default when no engine is configured.
    """

    # old_logp is the rollout (SGLang) log-prob, which is frozen on the segment
    # and does NOT change across mini-batch updates — so reusing it across
    # num_updates_per_batch>1 is the deliberate rollout-anchored PPO ratio
    # (verl bypass_mode=True parity), matching DRPO. The ratio then absorbs the
    # rollout-vs-train engine gap on later mini-batches (accepted for parity).
    supports_multi_update = True

    def __init__(
        self,
        *,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "ar",
        clip_range: float = 1e-4,
        clip_schedule: str = "constant",
        clip_range_high: Optional[float] = None,
        loss_agg_mode: str = "token-mean",
        horizon: int = 8192,
        conditions_cls: Optional[Type[Any]] = None,
        sampling_temperature: Optional[float] = None,
        tis_clip: Optional[float] = None,
        tis_level: str = "token",
        tis_mode: str = "truncate",
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError("GRPO: either `stage` or `pipeline` must be provided")
        if stage is None:
            stage = getattr(pipeline, stage_attr)
        self.stage = stage
        self.clip_range = float(clip_range)
        self.clip_range_high = None if clip_range_high is None else float(clip_range_high)
        self.clip_schedule = str(clip_schedule)
        self.loss_agg_mode = str(loss_agg_mode)
        self.horizon = int(horizon)
        self.conditions_cls = conditions_cls
        # Truncated Importance Sampling (verl rollout_is). DEFAULT OFF.
        # GRPO is always rollout-anchored (segment.log_probs IS the rollout logp),
        # so the rollout logp is always available and slice-aligned for TIS.
        self.tis_clip = None if tis_clip is None else float(tis_clip)
        self.tis_level = _validate_tis_level(tis_level)
        self.tis_mode = _validate_tis_mode(tis_mode)
        if sampling_temperature is None:
            from unirl.types.sampling import ARSamplingParams

            sampling_temperature = ARSamplingParams.__dataclass_fields__["temperature"].default
        self.sampling_temperature = float(sampling_temperature)

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        if segment.tokens is None or segment.lengths is None or segment.log_probs is None:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        if int(segment.tokens.shape[0]) == 0:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        new_logp = self.stage.replay(
            typed_conds, segment=segment, temperature=self.sampling_temperature
        )  # [total_tokens]
        # old_logp = the rollout log-prob, frozen on the segment — the deliberate
        # rollout-anchored ratio across num_updates_per_batch steps (see the
        # supports_multi_update class comment; verl bypass_mode=True parity).
        old_logp = segment.log_probs.to(dtype=new_logp.dtype, device=new_logp.device)
        adv_per_token = self._expand_advantages_to_tokens(
            advantages, segment.lengths, dtype=new_logp.dtype, device=new_logp.device
        )

        clip_range = _resolve_clip_range_from_schedule(self.clip_range, self.clip_schedule, training_progress)
        clip_high = (
            None
            if self.clip_range_high is None
            else _resolve_clip_range_from_schedule(self.clip_range_high, self.clip_schedule, training_progress)
        )
        loss_per_elem, ratio_metrics = _grpo_clip_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            advantages=adv_per_token,
            clip_range=clip_range,
            clip_range_high=clip_high,
        )

        # Truncated Importance Sampling (TIS, verl rollout_is). OFF unless
        # tis_clip is set. GRPO is rollout-anchored, so old_logp IS the rollout
        # logp (μ); TIS truncates the rollout-vs-train weight π_train/μ one-sidedly
        # and multiplies it (detached) onto the per-token loss — a guard ON TOP OF
        # the (two-sided) PPO clip. See docs/tis.md for why this adds only a
        # one-sided truncation beyond UniRL's already-rollout-anchored ratio.
        tis_metrics: Dict[str, Any] = {}
        if self.tis_clip is not None:
            tis_w, tis_m = tis_correction(
                new_logp=new_logp,
                rollout_logp=old_logp,
                lengths=segment.lengths.to(device=new_logp.device),
                tis_clip=self.tis_clip,
                tis_level=self.tis_level,
                tis_mode=self.tis_mode,
            )
            loss_per_elem = loss_per_elem * tis_w
            tis_metrics = {k: float(v.item()) for k, v in tis_m.items()}

        # Loss aggregation (match DRPO / verl loss_agg_mode):
        #  - "seq-mean-token-sum-norm" (Dr.GRPO/DAPO): per-seq token-SUM / horizon,
        #    then mean over sequences (length-UNbiased).
        #  - "seq-mean-token-mean" (ORIGINAL GRPO): per-seq token-MEAN, then mean
        #    over sequences (length-normalized, the standard-GRPO length bias).
        #  - "token-mean" (default): flat mean over all tokens.
        if self.loss_agg_mode in ("seq-mean-token-sum-norm", "seq-mean-token-mean") and segment.lengths is not None:
            parts = torch.split(loss_per_elem, segment.lengths.tolist())
            if self.loss_agg_mode == "seq-mean-token-sum-norm":
                loss = torch.stack([p.sum() for p in parts]).mean() / float(self.horizon)
            else:  # seq-mean-token-mean — guard 0-length responses (mean of empty = NaN)
                loss = torch.stack([p.mean() if p.numel() else p.new_zeros(()) for p in parts]).mean()
        else:
            loss = loss_per_elem.mean()
        (loss * loss_scale).backward()

        metrics: Dict[str, Any] = {
            "policy_loss": float(loss.detach().item()),
            "clip_range": float(clip_range),
            **rollout_replay_logp_absdiff(new_logp, old_logp),
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
            **tis_metrics,
        }
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=int(new_logp.shape[0]),
            has_backward=True,
        )

    @staticmethod
    def _expand_advantages_to_tokens(
        advantages: torch.Tensor,
        lengths: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Expand per-sample ``advantages [B]`` to per-token ``[total_tokens]``.

        Each sample's advantage is repeated across its ``lengths``-defined
        token span so that token positions in segment ``k`` all see
        ``advantages[k]``. ``lengths`` comes from
        :attr:`Batch.lengths` on the segment (derived from the framework-
        managed cu_seqlens).
        """
        bs = int(advantages.shape[0])
        if int(lengths.shape[0]) != bs:
            raise ValueError(f"GRPO advantage expansion: advantages batch={bs} != lengths={int(lengths.shape[0])}")
        chunks: List[torch.Tensor] = []
        adv_cast = advantages.detach().to(dtype=dtype, device=device)
        for k in range(bs):
            n = int(lengths[k].item())
            if n > 0:
                chunks.append(adv_cast[k].expand(n))
        if not chunks:
            return torch.zeros(0, dtype=dtype, device=device)
        return torch.cat(chunks, dim=0)


__all__ = ["GRPO", "GRPOConfig"]
