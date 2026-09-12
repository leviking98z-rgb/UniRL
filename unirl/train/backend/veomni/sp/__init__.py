"""Ulysses sequence-parallel (SP) setup shared by native and VeOmni FSDP."""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


def initialize_sequence_parallel_state(sp_size: int, *, device: torch.device) -> None:
    """Initialize and validate VeOmni's contiguous Ulysses process groups."""
    if sp_size <= 1:
        return

    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("sequence parallelism requires an initialized torch.distributed process group")
    world = dist.get_world_size()
    rank = dist.get_rank()
    if world % sp_size:
        raise ValueError(f"world_size {world} is not divisible by sp_size {sp_size}")

    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state

    init_parallel_state(
        dp_size=world // sp_size,
        ulysses_size=sp_size,
        dp_mode="fsdp2",
        device_type=device.type,
    )
    state = get_parallel_state()
    observed = {
        "dp_size": int(state.dp_size),
        "dp_replicate_size": int(state.dp_replicate_size),
        "dp_shard_size": int(state.dp_shard_size),
        "ulysses_size": int(state.ulysses_size),
        "cp_size": int(state.cp_size),
        "tp_size": int(state.tp_size),
        "pp_size": int(state.pp_size),
        "dp_mode": str(state.dp_mode),
        "include_sp_in_fsdp": bool(state.include_sp_in_fsdp),
        "device_type": str(state.device_type),
    }
    expected = {
        "dp_size": world // sp_size,
        "dp_replicate_size": 1,
        "dp_shard_size": world // sp_size,
        "ulysses_size": sp_size,
        "cp_size": 1,
        "tp_size": 1,
        "pp_size": 1,
        "dp_mode": "fsdp2",
        "include_sp_in_fsdp": True,
        "device_type": device.type,
    }
    if observed != expected:
        raise RuntimeError(
            "sequence parallelism found an incompatible pre-existing VeOmni parallel state: "
            f"expected {expected}, observed {observed}"
        )

    group = state.sp_group
    if group is None or dist.get_world_size(group) != sp_size or dist.get_rank(group) != rank % sp_size:
        raise RuntimeError(
            "sequence parallelism did not create the contiguous SP group required by UniRL's Handle layout "
            f"(rank={rank}, world={world}, sp_size={sp_size})"
        )
    get_group_ranks = getattr(dist, "get_process_group_ranks", None)
    if callable(get_group_ranks):
        actual_ranks = list(get_group_ranks(group))
        start = rank // sp_size * sp_size
        expected_ranks = list(range(start, start + sp_size))
        if actual_ranks != expected_ranks:
            raise RuntimeError(
                "sequence parallelism group does not match UniRL's contiguous Handle layout: "
                f"expected ranks {expected_ranks}, observed {actual_ranks}"
            )


def apply_sequence_parallelism(model: nn.Module, sp_size: int) -> None:
    """Install the Ulysses SP patch on ``model`` in place (no-op if sp_size<=1)."""
    if sp_size <= 1:
        return

    from unirl.train.backend.veomni.sp import ar, diffusion

    if ar.is_ar_causal_lm(model):
        ar.apply_ar_sequence_parallelism(model, sp_size)
        return

    if diffusion.is_diffusers_transformer(model):
        diffusion.apply_diffusion_sequence_parallelism(model, sp_size)
        return

    raise NotImplementedError(
        f"apply_sequence_parallelism: no SP patcher for {type(model).__name__} "
        "(neither an HF causal-LM nor a diffusers transformer)."
    )


__all__ = ["apply_sequence_parallelism", "initialize_sequence_parallel_state"]
