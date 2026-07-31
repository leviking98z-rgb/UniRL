"""Shared runtime for standard single-stream diffusion stages.

Model stages keep ownership of latent geometry, conditioning, guidance, and
the per-step model kernel. This runner owns the model-neutral sampling and
replay bookkeeping around that kernel. Multi-stream models such as LTX-2 keep
their specialized runtime because their video and audio states advance
together.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch

from unirl.models.types.replay_result import ReplayResult
from unirl.sde.kernels import StepStrategy
from unirl.sde.noise import generate_latents
from unirl.types.sampling import DiffusionSamplingParams, compute_trajectory_positions
from unirl.types.segments.latent import LatentSegment

Transition = Callable[..., Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]
SegmentFactory = Callable[..., LatentSegment]


class SingleStreamDiffusionRunner:
    """Run sampling and replay for one latent stream.

    ``transition`` is a model-stage-owned callable, normally a
    ``functools.partial`` around ``DiffusionStep.step_with_logp``. The runner
    supplies only model-neutral transition arguments.
    """

    def __init__(
        self,
        *,
        strategy: StepStrategy,
        autocast_dtype: torch.dtype,
        trajectory_dtype: torch.dtype,
        logprob_dtype: torch.dtype,
        owner: str,
    ) -> None:
        self.strategy = strategy
        self.autocast_dtype = autocast_dtype
        self.trajectory_dtype = trajectory_dtype
        self.logprob_dtype = logprob_dtype
        self.owner = str(owner)

    def _autocast(self, device: torch.device):
        if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16):
            return torch.autocast("cuda", self.autocast_dtype)
        return nullcontext()

    def sample(
        self,
        *,
        schedule: torch.Tensor,
        params: DiffusionSamplingParams,
        batch_size: int,
        latent_shape: Tuple[int, ...],
        device: torch.device,
        initial_latents: Optional[torch.Tensor],
        transition: Transition,
        segment_factory: SegmentFactory,
        shape_description: str = "",
    ) -> LatentSegment:
        """Initialize one latent stream, run its schedule, and pack a segment."""
        num_steps = int(params.num_inference_steps)
        if int(schedule.shape[0]) != num_steps + 1:
            raise ValueError(f"{self.owner}.diffuse: schedule length {schedule.shape[0]} != T+1={num_steps + 1}")
        schedule = schedule.to(device)
        self.strategy.init_schedule(schedule)

        expected_shape = tuple(int(dim) for dim in latent_shape)
        if initial_latents is not None:
            if int(initial_latents.shape[0]) != int(batch_size):
                raise ValueError(
                    f"{self.owner}.diffuse: initial_latents.shape[0]="
                    f"{int(initial_latents.shape[0])} != batch_size={int(batch_size)}."
                )
            if tuple(initial_latents.shape[1:]) != expected_shape:
                suffix = f" {shape_description}" if shape_description else ""
                raise ValueError(
                    f"{self.owner}.diffuse: initial_latents.shape[1:]="
                    f"{tuple(initial_latents.shape[1:])} != expected {expected_shape}{suffix}."
                )
            latents = initial_latents.to(device=device, dtype=self.trajectory_dtype)
        else:
            latents = generate_latents(
                batch_size=int(batch_size),
                latent_shape=expected_shape,
                device=device,
                dtype=self.trajectory_dtype,
                init_same_noise=bool(params.init_same_noise),
                samples_per_prompt=int(params.samples_per_prompt),
                noise_group_ids=params.noise_group_ids,
                base_seed=int(params.seed),
            )

        sde_set = {int(index) for index in (params.sde_indices or [])}
        needed = set(compute_trajectory_positions(sde_set, num_steps))
        needed.add(num_steps)

        stored_pairs: List[Tuple[int, torch.Tensor]] = []
        if 0 in needed:
            stored_pairs.append((0, latents.detach().clone()))
        sde_log_probs: List[torch.Tensor] = []
        sigma_max = float(schedule[1].item()) if int(schedule.shape[0]) > 1 else 0.99

        for step_index in range(num_steps):
            sigma = schedule[step_index].to(device)
            sigma_next = schedule[step_index + 1].to(device)
            step_eta = float(params.eta) if step_index in sde_set else 0.0
            with torch.no_grad(), self._autocast(device):
                latents, log_prob, _ = transition(
                    sample=latents,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    eta=step_eta,
                    sigma_max=sigma_max,
                    step_index=step_index,
                )
            latents = latents.to(dtype=self.trajectory_dtype)
            if (step_index + 1) in needed:
                stored_pairs.append((step_index + 1, latents.detach().clone()))
            if log_prob is not None:
                sde_log_probs.append(log_prob.to(dtype=self.logprob_dtype))

        positions = [position for position, _ in stored_pairs]
        return segment_factory(
            latents=torch.stack([value for _, value in stored_pairs], dim=1),
            sigmas=schedule,
            indices=torch.tensor(positions, dtype=torch.long, device=device),
            sde_logp=torch.stack(sde_log_probs, dim=1) if sde_log_probs else None,
            sde_indices=(torch.tensor(sorted(sde_set), dtype=torch.long, device=device) if sde_set else None),
        )

    def replay(
        self,
        *,
        segment: LatentSegment,
        params: DiffusionSamplingParams,
        transition: Transition,
        step_indices: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
        expected_latents_ndim: Optional[int] = None,
        latent_layout: str = "",
    ) -> ReplayResult:
        """Replay stored SDE transitions through the same model-step callable."""
        if segment.sde_indices is None or segment.latents is None:
            raise ValueError(f"{self.owner}.replay: segment.sde_indices / latents missing")
        if segment.sigmas is None:
            raise ValueError(f"{self.owner}.replay: segment.sigmas missing")
        if expected_latents_ndim is not None and segment.latents.ndim != expected_latents_ndim:
            expected = f" {latent_layout}" if latent_layout else ""
            raise ValueError(f"{self.owner}.replay: expected latents{expected}, got {tuple(segment.latents.shape)}")

        available = {int(index) for index in segment.sde_indices.tolist()}
        target = (
            [int(index) for index in step_indices]
            if step_indices is not None
            else [int(index) for index in segment.sde_indices.tolist()]
        )
        unknown = [index for index in target if index not in available]
        if unknown:
            raise ValueError(
                f"{self.owner}.replay: step_indices {unknown} not in segment.sde_indices={sorted(available)}"
            )

        replay_device = torch.device(device) if device is not None else segment.latents.device
        sigmas = segment.sigmas.to(replay_device)
        sigma_max = float(sigmas[1].item()) if int(sigmas.shape[0]) > 1 else 0.99
        log_probs: List[torch.Tensor] = []
        prev_sample_means: List[torch.Tensor] = []

        with self._autocast(replay_device):
            for step_index in target:
                _, log_prob, prev_mean = transition(
                    sample=segment.latents_at(step_index),
                    prev_sample=segment.latents_at(step_index + 1),
                    sigma=sigmas[step_index].to(dtype=torch.float32),
                    sigma_next=sigmas[step_index + 1].to(dtype=torch.float32),
                    eta=float(params.eta),
                    sigma_max=sigma_max,
                    step_index=step_index,
                )
                if log_prob is None:
                    raise RuntimeError(
                        f"{self.owner}.replay: strategy returned None log-prob "
                        f"at step_index={step_index} (deterministic mode); replay "
                        "requires a stochastic SDE strategy."
                    )
                log_probs.append(log_prob)
                if prev_mean is not None:
                    prev_sample_means.append(prev_mean)

        return ReplayResult(
            log_probs=torch.stack(log_probs, dim=1).to(dtype=self.logprob_dtype),
            prev_sample_means=(
                torch.stack(prev_sample_means, dim=1).to(dtype=self.trajectory_dtype) if prev_sample_means else None
            ),
        )


__all__ = ["SingleStreamDiffusionRunner"]
