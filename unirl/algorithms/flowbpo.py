"""Bellman Policy Optimization variants for stochastic diffusion policies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import torch

from unirl.config.require import require
from unirl.distributed.tensor.ref import hydrate
from unirl.types.conditions import Condition
from unirl.types.sample import Part
from unirl.types.segments.latent import LatentSegment

from .base import (
    AlgorithmStepResult,
    BaseAlgorithmConfig,
    StageAlgorithm,
    _gaussian_kl_div,
    _reference_kl_loss,
    _reference_replay_means,
    _resolve_reference_model,
    _transition_sigma,
    gather_sde_field,
    typed_conditions,
)


@dataclass
class FlowBPOConfig(BaseAlgorithmConfig):
    stage_attr: str = "diffusion"
    conditions_cls: str = ""
    bpo_eta: float = 0.1
    reward_epsilon: float = 1e-8
    beta: float = 0.0
    params: Any = dc_field(default=None)


@dataclass
class FlowBPOFullKLConfig(BaseAlgorithmConfig):
    """Configuration for the continuous-Gaussian form of BPO Equation 26."""

    stage_attr: str = "diffusion"
    conditions_cls: str = ""
    beta: float = 0.0
    params: Any = dc_field(default=None)


def _group_reward_targets(
    rewards: torch.Tensor,
    group_ids: Sequence[str],
    *,
    expected_group_size: int,
    reward_epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return group-centered rewards and inverse group standard deviations ``[B]``."""
    rewards = rewards.to(torch.float32).reshape(-1)
    if rewards.numel() != len(group_ids):
        raise ValueError(f"FlowBPO: rewards count {rewards.numel()} != group_ids count {len(group_ids)}.")

    rows_by_group: Dict[str, List[int]] = {}
    for row, group_id in enumerate(group_ids):
        rows_by_group.setdefault(str(group_id), []).append(row)
    bad_sizes = {group_id: len(rows) for group_id, rows in rows_by_group.items() if len(rows) != expected_group_size}
    if bad_sizes:
        raise ValueError(
            "FlowBPO requires every data-parallel shard to contain complete sibling groups; "
            f"expected {expected_group_size} samples per prompt, got {bad_sizes}. "
            "Choose batch_size so prompt groups do not cross DP shard boundaries."
        )

    residual = torch.zeros_like(rewards)
    weight = torch.zeros_like(rewards)
    for rows in rows_by_group.values():
        index = torch.tensor(rows, dtype=torch.long, device=rewards.device)
        values = rewards.index_select(0, index)
        finite = torch.isfinite(values)
        count = int(finite.sum().item())
        if count == 0:
            continue
        finite_values = torch.where(finite, values, torch.zeros_like(values))
        mean = finite_values.sum() / count
        centered = torch.where(finite, values - mean, torch.zeros_like(values))
        variance = centered.square().sum() / count
        std = variance.sqrt() if count > 1 else torch.ones_like(variance)
        residual.index_copy_(0, index, centered)
        weight.index_copy_(0, index, finite.to(values.dtype) / (std + reward_epsilon))
    return residual, weight


def _flowbpo_residual_loss(
    *,
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    new_means: torch.Tensor,
    old_means: torch.Tensor,
    actions: torch.Tensor,
    sigma_t: torch.Tensor,
    reward_residual: torch.Tensor,
    reward_weight: torch.Tensor,
    bpo_eta: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the mean-reduced Gaussian FlowBPO trajectory residual loss."""
    latent_dims = tuple(range(2, new_means.ndim))
    delta_mean = new_means.float() - old_means.float()
    sigma_sq = sigma_t.float().square()
    q_step = ((actions.float() - old_means.float()) * delta_mean / sigma_sq).mean(dim=latent_dims)
    trajectory_correction = q_step.sum(dim=1)

    reward_residual = reward_residual.detach().to(device=new_means.device, dtype=torch.float32).reshape(-1)
    reward_weight = reward_weight.detach().to(device=new_means.device, dtype=torch.float32).reshape(-1)
    bellman_residual = float(bpo_eta) * reward_residual - trajectory_correction
    loss_per_sample = reward_weight * bellman_residual.square() / (2.0 * float(bpo_eta))
    loss = loss_per_sample.mean()

    with torch.no_grad():
        kl_step = _gaussian_kl_div(new_means.float(), old_means.float(), sigma_t.float()).mean(dim=latent_dims)
        q_from_logp = new_logp.float() - old_logp.float() + kl_step
        identity_error = (q_from_logp - q_step.detach()).abs()
        q_std = (
            trajectory_correction.detach().std()
            if trajectory_correction.numel() > 1
            else torch.zeros((), device=trajectory_correction.device)
        )
        residual_rms = (reward_weight * bellman_residual.detach().square()).mean().sqrt()
        metrics = {
            "bpo_q_mean": trajectory_correction.detach().mean(),
            "bpo_q_std": q_std,
            "bpo_q_abs_max": trajectory_correction.detach().abs().max(),
            "bpo_q_step_abs_mean": q_step.detach().abs().mean(),
            "bpo_path_kl_mean": kl_step.sum(dim=1).mean(),
            "bpo_residual_rms": residual_rms,
            "bpo_identity_error_mean": identity_error.mean(),
            "bpo_identity_error_max": identity_error.max(),
            "bpo_reward_residual_std": reward_residual.std()
            if reward_residual.numel() > 1
            else reward_residual.abs().mean(),
            "bpo_reward_weight_mean": reward_weight.mean(),
            "bpo_reward_weight_max": reward_weight.max(),
            "bpo_valid_fraction": (reward_weight > 0).float().mean(),
        }
    return loss, metrics


def _flowbpo_full_kl_loss(
    *,
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    new_means: torch.Tensor,
    old_means: torch.Tensor,
    actions: torch.Tensor,
    sigma_t: torch.Tensor,
    advantages: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Return the BPO Equation 26 loss for equal-variance Gaussian transitions."""
    if new_logp.shape != old_logp.shape:
        raise ValueError(
            "FlowBPOFullKL: new_logp and old_logp must have identical shapes; "
            f"got {tuple(new_logp.shape)} and {tuple(old_logp.shape)}."
        )
    if new_means.shape != old_means.shape or new_means.shape != actions.shape:
        raise ValueError(
            "FlowBPOFullKL: new_means, old_means, and actions must have identical shapes; "
            f"got {tuple(new_means.shape)}, {tuple(old_means.shape)}, and {tuple(actions.shape)}."
        )

    latent_dims = tuple(range(2, new_means.ndim))
    new_means_f = new_means.float()
    old_means_f = old_means.detach().float()
    actions_f = actions.detach().float()
    sigma_sq = sigma_t.float().square()

    q_step = ((actions_f - old_means_f) * (new_means_f - old_means_f) / sigma_sq).mean(dim=latent_dims)
    adv = advantages.detach().to(device=q_step.device, dtype=q_step.dtype).reshape(-1, 1)
    if adv.shape[0] != q_step.shape[0]:
        raise ValueError(
            "FlowBPOFullKL: advantages batch does not match replay batch; "
            f"got {adv.shape[0]} and {q_step.shape[0]}."
        )
    loss_per_step = -adv * q_step
    loss = loss_per_step.mean()

    with torch.no_grad():
        reverse_kl_step = _gaussian_kl_div(old_means_f, new_means_f, sigma_t.float()).mean(dim=latent_dims)
        log_ratio = new_logp.float() - old_logp.detach().float()
        q_from_logp = log_ratio + reverse_kl_step
        identity_error = (q_from_logp - q_step.detach()).abs()
        metrics = {
            "bpo_q_mean": q_step.detach().mean(),
            "bpo_q_std": q_step.detach().std()
            if q_step.numel() > 1
            else torch.zeros((), device=q_step.device),
            "bpo_q_abs_max": q_step.detach().abs().max(),
            "bpo_reverse_kl_mean": reverse_kl_step.mean(),
            "bpo_reverse_kl_max": reverse_kl_step.max(),
            "bpo_log_ratio_mean": log_ratio.mean(),
            "bpo_identity_error_mean": identity_error.mean(),
            "bpo_identity_error_max": identity_error.max(),
            "bpo_adv_mean": adv.mean(),
            "bpo_adv_std": adv.std() if adv.numel() > 1 else torch.zeros((), device=adv.device),
        }
    return loss, metrics


class FlowBPO(StageAlgorithm):
    """Trajectory-level Bellman residual optimization for stochastic diffusion policies."""

    supports_multi_update = True
    requires_backend = True
    recomputes_anchor = True
    anchor_fields = ("sde_logp", "sde_means")

    def __init__(
        self,
        *,
        params: Any,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "diffusion",
        bpo_eta: float = 0.1,
        reward_epsilon: float = 1e-8,
        beta: float = 0.0,
        backend: Any = None,
        conditions_cls: Optional[Type[Any]] = None,
    ) -> None:
        if stage is None and pipeline is not None:
            stage = getattr(pipeline, stage_attr)
        if stage is None:
            raise ValueError("FlowBPO: either `stage` or `pipeline` must be provided")
        self.stage = stage
        self.params = params
        self.bpo_eta = float(bpo_eta)
        self.reward_epsilon = float(reward_epsilon)
        self.beta = float(beta)
        require(
            math.isfinite(self.bpo_eta) and self.bpo_eta > 0.0,
            f"FlowBPO: bpo_eta must be finite and > 0; got {bpo_eta}.",
        )
        require(
            math.isfinite(self.reward_epsilon) and self.reward_epsilon > 0.0,
            f"FlowBPO: reward_epsilon must be finite and > 0; got {reward_epsilon}.",
        )
        require(float(self.params.eta) > 0.0, "FlowBPO requires stochastic SDE sampling with params.eta > 0.")
        self._ref_model = _resolve_reference_model(backend, beta=self.beta, algo="FlowBPO")
        self.conditions_cls = conditions_cls

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: LatentSegment,
    ) -> None:
        """Freeze old-policy transition log-probs and means before optimizer updates."""
        target_steps = self._resolve_target_steps(segment)
        if not target_steps:
            return
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        with torch.no_grad():
            result = self.stage.replay(typed_conds, segment=segment, params=self.params, step_indices=target_steps)
        if result.prev_sample_means is None:
            raise RuntimeError("FlowBPO requires stage.replay() to return prev_sample_means.")
        segment.sde_logp = result.log_probs.detach().cpu()
        segment.sde_means = result.prev_sample_means.detach().cpu()

    def prepare_part(self, part: Part) -> Part:
        """Attach exact group reward residuals after the old-policy anchor is assembled."""
        if part.segment is None or not self._resolve_target_steps(part.segment):
            return part
        if part.rewards is None:
            raise ValueError("FlowBPO requires terminal rewards on the generated Part.")
        if part.sampling_params is None:
            raise ValueError("FlowBPO requires sampling_params to validate complete prompt groups.")
        expected_group_size = int(part.sampling_params.samples_per_prompt)
        require(expected_group_size > 1, "FlowBPO requires samples_per_prompt > 1 to estimate V^mu.")
        reward_residual, reward_weight = _group_reward_targets(
            hydrate(part.rewards),
            part.group_ids,
            expected_group_size=expected_group_size,
            reward_epsilon=self.reward_epsilon,
        )
        part.segment.bpo_reward_residual = reward_residual.cpu()
        part.segment.bpo_reward_weight = reward_weight.cpu()
        return part

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: LatentSegment,
        advantages: Optional[torch.Tensor],
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        del advantages, training_progress
        target_steps = self._resolve_target_steps(segment)
        if not target_steps:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        if segment.bpo_reward_residual is None or segment.bpo_reward_weight is None:
            raise RuntimeError("FlowBPO reward targets were not prepared before the micro-batch update.")

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        replay_result = self.stage.replay(
            typed_conds,
            segment=segment,
            params=self.params,
            step_indices=target_steps,
        )
        new_logp = replay_result.log_probs
        new_means = replay_result.prev_sample_means
        if new_means is None:
            raise RuntimeError("FlowBPO requires stage.replay() to return prev_sample_means.")

        old_logp = gather_sde_field(segment.sde_logp, segment.sde_indices, target_steps, field_name="sde_logp").to(
            dtype=new_logp.dtype, device=new_logp.device
        )
        old_means = gather_sde_field(segment.sde_means, segment.sde_indices, target_steps, field_name="sde_means").to(
            dtype=new_means.dtype, device=new_means.device
        )
        actions = torch.stack([segment.latents_at(step + 1) for step in target_steps], dim=1).to(new_means.device)
        sigma_t = _transition_sigma(
            self.stage,
            segment=segment,
            target_steps=target_steps,
            eta=float(self.params.eta),
            device=new_means.device,
            add_coefficient=True,
        )

        policy_loss, tensor_metrics = _flowbpo_residual_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            new_means=new_means,
            old_means=old_means,
            actions=actions,
            sigma_t=sigma_t,
            reward_residual=segment.bpo_reward_residual,
            reward_weight=segment.bpo_reward_weight,
            bpo_eta=self.bpo_eta,
        )
        loss = policy_loss
        metrics: Dict[str, Any] = {
            "policy_loss": float(policy_loss.detach().item()),
            "bpo_eta": self.bpo_eta,
            **{name: float(value.item()) for name, value in tensor_metrics.items()},
        }

        if self.beta > 0.0:
            ref_means = _reference_replay_means(
                self.stage,
                self._ref_model,
                conditions=typed_conds,
                segment=segment,
                params=self.params,
                target_steps=target_steps,
            ).to(dtype=new_means.dtype, device=new_means.device)
            kl_ref = _reference_kl_loss(new_means, ref_means, sigma_t)
            loss = loss + self.beta * kl_ref
            metrics["beta"] = self.beta
            metrics["kl_ref_mean"] = float(kl_ref.detach().item())

        (loss * loss_scale).backward()
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=len(target_steps),
            has_backward=True,
        )

    def _resolve_target_steps(self, segment: LatentSegment) -> List[int]:
        """Return all SDE-recorded trajectory step indices."""
        if segment.sde_indices is None:
            return []
        return [int(index) for index in segment.sde_indices.tolist()]


class FlowBPOFullKL(StageAlgorithm):
    """Paper Equation 26 with exact reverse KL for Gaussian SDE transitions."""

    supports_multi_update = True
    requires_backend = True
    recomputes_anchor = True
    anchor_fields = ("sde_logp", "sde_means")

    def __init__(
        self,
        *,
        params: Any,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "diffusion",
        beta: float = 0.0,
        backend: Any = None,
        conditions_cls: Optional[Type[Any]] = None,
    ) -> None:
        if stage is None and pipeline is not None:
            stage = getattr(pipeline, stage_attr)
        if stage is None:
            raise ValueError("FlowBPOFullKL: either `stage` or `pipeline` must be provided")
        self.stage = stage
        self.params = params
        self.beta = float(beta)
        require(float(self.params.eta) > 0.0, "FlowBPOFullKL requires stochastic SDE sampling with params.eta > 0.")
        self._ref_model = _resolve_reference_model(backend, beta=self.beta, algo="FlowBPOFullKL")
        self.conditions_cls = conditions_cls

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: LatentSegment,
    ) -> None:
        """Freeze rollout-policy log probabilities and means before optimizer updates."""
        target_steps = self._resolve_target_steps(segment)
        if not target_steps:
            return
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        with torch.no_grad():
            result = self.stage.replay(typed_conds, segment=segment, params=self.params, step_indices=target_steps)
        if result.prev_sample_means is None:
            raise RuntimeError("FlowBPOFullKL requires stage.replay() to return prev_sample_means.")
        segment.sde_logp = result.log_probs.detach().cpu()
        segment.sde_means = result.prev_sample_means.detach().cpu()

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: LatentSegment,
        advantages: Optional[torch.Tensor],
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        del training_progress
        target_steps = self._resolve_target_steps(segment)
        if not target_steps:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        if advantages is None:
            raise ValueError("FlowBPOFullKL requires group-normalized advantages.")

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        replay_result = self.stage.replay(
            typed_conds,
            segment=segment,
            params=self.params,
            step_indices=target_steps,
        )
        new_logp = replay_result.log_probs
        new_means = replay_result.prev_sample_means
        if new_means is None:
            raise RuntimeError("FlowBPOFullKL requires stage.replay() to return prev_sample_means.")

        old_logp = gather_sde_field(segment.sde_logp, segment.sde_indices, target_steps, field_name="sde_logp").to(
            dtype=new_logp.dtype, device=new_logp.device
        )
        old_means = gather_sde_field(segment.sde_means, segment.sde_indices, target_steps, field_name="sde_means").to(
            dtype=new_means.dtype, device=new_means.device
        )
        actions = torch.stack([segment.latents_at(step + 1) for step in target_steps], dim=1).to(new_means.device)
        sigma_t = _transition_sigma(
            self.stage,
            segment=segment,
            target_steps=target_steps,
            eta=float(self.params.eta),
            device=new_means.device,
            add_coefficient=True,
        )

        policy_loss, tensor_metrics = _flowbpo_full_kl_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            new_means=new_means,
            old_means=old_means,
            actions=actions,
            sigma_t=sigma_t,
            advantages=advantages,
        )
        loss = policy_loss
        metrics: Dict[str, Any] = {
            "policy_loss": float(policy_loss.detach().item()),
            **{name: float(value.item()) for name, value in tensor_metrics.items()},
        }

        if self.beta > 0.0:
            ref_means = _reference_replay_means(
                self.stage,
                self._ref_model,
                conditions=typed_conds,
                segment=segment,
                params=self.params,
                target_steps=target_steps,
            ).to(dtype=new_means.dtype, device=new_means.device)
            kl_ref = _reference_kl_loss(new_means, ref_means, sigma_t)
            loss = loss + self.beta * kl_ref
            metrics["beta"] = self.beta
            metrics["kl_ref_mean"] = float(kl_ref.detach().item())

        (loss * loss_scale).backward()
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=len(target_steps),
            has_backward=True,
        )

    def _resolve_target_steps(self, segment: LatentSegment) -> List[int]:
        """Return all SDE-recorded trajectory step indices."""
        if segment.sde_indices is None:
            return []
        return [int(index) for index in segment.sde_indices.tolist()]


__all__ = ["FlowBPO", "FlowBPOConfig", "FlowBPOFullKL", "FlowBPOFullKLConfig"]
