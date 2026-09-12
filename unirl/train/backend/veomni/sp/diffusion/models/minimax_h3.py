"""SP adapter for MiniMaxH3Transformer3DModel."""

from __future__ import annotations

import importlib
import logging
from typing import Any

import torch
from torch import Tensor, nn

from unirl.train.backend.veomni.sp.diffusion.ulysses import (
    _sp,
    register,
    register_attention_installer,
)

logger = logging.getLogger(__name__)

_MODEL_ARG_POSITIONS = {
    "timestep_indices": 4,
    "token_tags": 5,
    "position_ids": 6,
}
_BLOCK_ARG_POSITIONS = {
    "hidden_states": 0,
    "adaln_indices": 2,
    "rotary_emb": 3,
    "attention_mask": 4,
}
_NORM_ARG_POSITIONS = {
    "hidden_states": 0,
    "timestep_indices": 2,
}
_MISSING = object()


def _read_arg(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
    position: int,
    default: Any = _MISSING,
) -> Any:
    if name in kwargs:
        return kwargs[name]
    if len(args) > position:
        return args[position]
    if default is not _MISSING:
        return default
    raise ValueError(f"MiniMax-H3 SP requires forward argument {name!r}")


def _write_arg(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
    position: int,
    value: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if name in kwargs:
        kwargs[name] = value
        return args, kwargs
    new_args = list(args)
    if len(new_args) <= position:
        raise ValueError(f"MiniMax-H3 SP cannot replace missing forward argument {name!r}")
    new_args[position] = value
    return tuple(new_args), kwargs


def _slice_rows(tensor: Tensor, *, dim: int, group: Any) -> Tensor:
    return _sp().slice_input_tensor(tensor, dim=dim, padding=False, group=group)


class MiniMaxH3SPAttnProcessor:
    """MiniMax-H3 self-attention with Ulysses sequence/head all-to-all."""

    def __init__(self, sp_group: Any, original_processor: Any) -> None:
        self.sp_group = sp_group
        self.original_processor = original_processor
        self._attention_backend = getattr(original_processor, "_attention_backend", None)
        self._parallel_config = getattr(original_processor, "_parallel_config", None)
        module = importlib.import_module(type(original_processor).__module__)
        self._apply_rotary_emb = module._apply_rotary_emb
        self._dispatch_attention_fn = module.dispatch_attention_fn
        sp = _sp()
        self._gather_seq = sp.gather_seq_scatter_heads
        self._gather_heads = sp.gather_heads_scatter_seq

    def __call__(
        self,
        attn: nn.Module,
        hidden_states: Tensor,
        rotary_emb: tuple[Tensor, Tensor] | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))
        if rotary_emb is not None:
            query = self._apply_rotary_emb(query, *rotary_emb)
            key = self._apply_rotary_emb(key, *rotary_emb)

        query = self._gather_seq(query, seq_dim=1, head_dim=2, group=self.sp_group)
        key = self._gather_seq(key, seq_dim=1, head_dim=2, group=self.sp_group)
        value = self._gather_seq(value, seq_dim=1, head_dim=2, group=self.sp_group)
        hidden_states = self._dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = self._gather_heads(
            hidden_states,
            head_dim=2,
            seq_dim=1,
            group=self.sp_group,
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


@register_attention_installer("MiniMaxH3Transformer3DModel")
def _install_h3_attention(model: nn.Module, sp_group: Any) -> None:
    import torch.distributed as dist

    blocks = tuple(getattr(model, "transformer_blocks", ()))
    if not blocks:
        raise ValueError("MiniMax-H3 SP found no transformer_blocks")
    sp_size = dist.get_world_size(sp_group)
    for index, block in enumerate(blocks):
        attn = getattr(block, "attn", None)
        if attn is None or not hasattr(attn, "get_processor") or not hasattr(attn, "set_processor"):
            raise TypeError(f"MiniMax-H3 SP block {index} has no replaceable attention processor")
        if int(attn.heads) % sp_size:
            raise ValueError(
                f"MiniMax-H3 SP requires num_heads % sp_size == 0, but block {index} has "
                f"{attn.heads} heads and sp_size={sp_size}"
            )
        original = attn.get_processor()
        if isinstance(original, MiniMaxH3SPAttnProcessor):
            continue
        attn.set_processor(MiniMaxH3SPAttnProcessor(sp_group, original))
    logger.info("diffusion SP: installed MiniMax-H3 attention on %d main blocks", len(blocks))


@register("MiniMaxH3Transformer3DModel")
def _wrap_h3(model: nn.Module, sp_group: Any) -> None:
    import torch.distributed as dist

    if getattr(model, "_unirl_h3_sp_installed", False):
        return
    sp_size = dist.get_world_size(sp_group)
    state: dict[str, int] = {}

    def model_pre(_module, args, kwargs):
        args = tuple(args)
        kwargs = dict(kwargs)
        position_ids = _read_arg(args, kwargs, "position_ids", _MODEL_ARG_POSITIONS["position_ids"])
        token_tags = _read_arg(args, kwargs, "token_tags", _MODEL_ARG_POSITIONS["token_tags"])
        timestep_indices = _read_arg(
            args,
            kwargs,
            "timestep_indices",
            _MODEL_ARG_POSITIONS["timestep_indices"],
        )
        original_length = int(position_ids.shape[0])
        if token_tags.shape != (original_length,) or timestep_indices.shape != (original_length,):
            raise ValueError(
                "MiniMax-H3 SP requires position_ids, token_tags and timestep_indices to share one sequence length"
            )
        pad = (-original_length) % sp_size
        if pad:
            position_ids = torch.cat(
                [position_ids, position_ids.new_zeros((pad, position_ids.shape[1]))],
                dim=0,
            )
            token_tags = torch.cat([token_tags, token_tags.new_full((pad,), -1)], dim=0)
            timestep_indices = torch.cat(
                [timestep_indices, timestep_indices.new_zeros((pad,))],
                dim=0,
            )
            args, kwargs = _write_arg(
                args,
                kwargs,
                "position_ids",
                _MODEL_ARG_POSITIONS["position_ids"],
                position_ids,
            )
            args, kwargs = _write_arg(
                args,
                kwargs,
                "token_tags",
                _MODEL_ARG_POSITIONS["token_tags"],
                token_tags,
            )
            args, kwargs = _write_arg(
                args,
                kwargs,
                "timestep_indices",
                _MODEL_ARG_POSITIONS["timestep_indices"],
                timestep_indices,
            )
        state["original_length"] = original_length
        state["padded_length"] = original_length + pad
        return args, kwargs

    def block_pre(_module, args, kwargs):
        args = tuple(args)
        kwargs = dict(kwargs)
        hidden_states = _read_arg(
            args,
            kwargs,
            "hidden_states",
            _BLOCK_ARG_POSITIONS["hidden_states"],
        )
        adaln_indices = _read_arg(
            args,
            kwargs,
            "adaln_indices",
            _BLOCK_ARG_POSITIONS["adaln_indices"],
        )
        global_length = int(adaln_indices.shape[0])
        if global_length % sp_size:
            raise ValueError(f"MiniMax-H3 SP received unpadded sequence length {global_length} for sp_size={sp_size}")
        local_length = global_length // sp_size
        if hidden_states.shape[1] == global_length:
            hidden_states = _slice_rows(hidden_states, dim=1, group=sp_group)
        elif hidden_states.shape[1] != local_length:
            raise ValueError(
                f"MiniMax-H3 SP expected hidden sequence length {global_length} or {local_length}, "
                f"got {hidden_states.shape[1]}"
            )
        args, kwargs = _write_arg(
            args,
            kwargs,
            "hidden_states",
            _BLOCK_ARG_POSITIONS["hidden_states"],
            hidden_states,
        )

        if adaln_indices.shape[0] == global_length:
            adaln_indices = _slice_rows(adaln_indices, dim=0, group=sp_group)
        args, kwargs = _write_arg(
            args,
            kwargs,
            "adaln_indices",
            _BLOCK_ARG_POSITIONS["adaln_indices"],
            adaln_indices,
        )

        rotary_emb = _read_arg(args, kwargs, "rotary_emb", _BLOCK_ARG_POSITIONS["rotary_emb"])
        if rotary_emb is not None:
            if len(rotary_emb) != 2:
                raise ValueError("MiniMax-H3 SP expects rotary_emb=(cos, sin)")
            rotary_emb = tuple(
                _slice_rows(tensor, dim=0, group=sp_group) if tensor.shape[0] == global_length else tensor
                for tensor in rotary_emb
            )
            if any(tensor.shape[0] != local_length for tensor in rotary_emb):
                raise ValueError("MiniMax-H3 SP received rotary embeddings with an inconsistent sequence length")
            args, kwargs = _write_arg(
                args,
                kwargs,
                "rotary_emb",
                _BLOCK_ARG_POSITIONS["rotary_emb"],
                rotary_emb,
            )

        attention_mask = _read_arg(
            args,
            kwargs,
            "attention_mask",
            _BLOCK_ARG_POSITIONS["attention_mask"],
            None,
        )
        if attention_mask is not None and attention_mask.shape[-2:] != (global_length, global_length):
            raise ValueError(
                "MiniMax-H3 SP requires the full packed-sequence attention mask on every rank, "
                f"got {tuple(attention_mask.shape)} for sequence length {global_length}"
            )
        return args, kwargs

    def norm_out_pre(_module, args, kwargs):
        args = tuple(args)
        kwargs = dict(kwargs)
        hidden_states = _read_arg(
            args,
            kwargs,
            "hidden_states",
            _NORM_ARG_POSITIONS["hidden_states"],
        )
        padded_length = state.get("padded_length")
        original_length = state.get("original_length")
        if padded_length is None or original_length is None:
            raise RuntimeError("MiniMax-H3 SP output gather ran without model-boundary state")
        hidden_states = _sp().gather_outputs(hidden_states, gather_dim=1, group=sp_group)
        if hidden_states.shape[1] != padded_length:
            raise RuntimeError(f"MiniMax-H3 SP gathered {hidden_states.shape[1]} rows, expected {padded_length}")
        hidden_states = hidden_states[:, :original_length]
        args, kwargs = _write_arg(
            args,
            kwargs,
            "hidden_states",
            _NORM_ARG_POSITIONS["hidden_states"],
            hidden_states,
        )

        timestep_indices = _read_arg(
            args,
            kwargs,
            "timestep_indices",
            _NORM_ARG_POSITIONS["timestep_indices"],
        )
        if timestep_indices.shape[0] == padded_length:
            timestep_indices = timestep_indices[:original_length]
        if timestep_indices.shape != (original_length,):
            raise ValueError("MiniMax-H3 SP output timestep_indices do not match the unpadded sequence length")
        args, kwargs = _write_arg(
            args,
            kwargs,
            "timestep_indices",
            _NORM_ARG_POSITIONS["timestep_indices"],
            timestep_indices,
        )
        return args, kwargs

    model.register_forward_pre_hook(model_pre, with_kwargs=True)
    for block in model.transformer_blocks:
        block.register_forward_pre_hook(block_pre, with_kwargs=True)
    model.norm_out.register_forward_pre_hook(norm_out_pre, with_kwargs=True)
    model._unirl_h3_sp_installed = True
    logger.info(
        "diffusion SP: MiniMax-H3 boundary hooks installed (blocks=%d, sp_size=%d)",
        len(model.transformer_blocks),
        sp_size,
    )


__all__ = ["MiniMaxH3SPAttnProcessor"]
