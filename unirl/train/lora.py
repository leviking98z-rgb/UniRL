"""Plain LoRA adapter injection.

Build-time structural mutation only: :func:`inject_lora` installs a single
peft adapter on the trainable stage and stamps its post-materialize
initialization via ``unirl.train.deferred``. No Shadow, no EMA — the
dual-adapter NFT variant lives in ``unirl.train.ema``.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import partial
from typing import Iterator, Sequence

import torch
from torch import nn

from unirl.train.deferred import _stamp

logger = logging.getLogger(__name__)


def inject_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: int,
    target_modules: Sequence[str],
    dropout: float = 0.0,
    bias: str = "none",
    task_type: str = "FEATURE_EXTRACTION",
    adapter_name: str = "default",
    init_seed: int | None = None,
) -> None:
    """Inject a single LoRA adapter.  No Shadow, no EMA."""
    from peft import LoraConfig, inject_adapter_in_model

    peft_cfg = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=list(target_modules),
        bias=str(bias),
        task_type=str(task_type),
    )
    with _adapter_init_seed(init_seed):
        inject_adapter_in_model(peft_cfg, model, adapter_name=adapter_name)

    if _current_rank() == 0:
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        logger.info(
            "inject_lora: adapter %r (rank=%d, alpha=%d, target_modules=%s) — %d trainable params",
            adapter_name,
            rank,
            alpha,
            tuple(target_modules),
            n_trainable,
        )

    if init_seed is not None:
        initial_state = _capture_materialized_adapter(model, name=adapter_name)
        if initial_state is not None:
            # FSDP shards the already-initialized tensors in place. Restore the
            # rank-0 full tensors after wrapping instead of calling PEFT's reset
            # on each local DTensor shard: shard-local CPU/CUDA RNG streams are
            # neither engine-independent nor reproducible.
            _stamp(
                model,
                partial(
                    _restore_adapter,
                    name=adapter_name,
                    state_dict=initial_state,
                ),
            )
            return

    _stamp(
        model,
        partial(
            _reset_adapter,
            name=adapter_name,
            init_seed=init_seed,
        ),
    )


def _reset_adapter(model: nn.Module, *, name: str, init_seed: int | None = None) -> None:
    from peft.tuners.lora import LoraLayer

    n_reset = 0
    with _adapter_init_seed(init_seed):
        for m in model.modules():
            if isinstance(m, LoraLayer):
                m.reset_lora_parameters(name, init_lora_weights=True)
                n_reset += 1
    if _current_rank() == 0:
        logger.info("_reset_adapter(%r): %d LoraLayer(s)", name, n_reset)


def _capture_materialized_adapter(
    model: nn.Module,
    *,
    name: str,
) -> dict[str, torch.Tensor] | None:
    """Capture seeded eager LoRA tensors on rank 0 for post-FSDP restore.

    ``None`` means at least one adapter tensor is meta and must use the
    historical post-materialize reset fallback. Other ranks return an empty
    dict; the restore broadcasts rank 0's full tensors through DCP.
    """
    token_a = f".lora_A.{name}."
    token_b = f".lora_B.{name}."
    params = {key: param for key, param in model.named_parameters() if token_a in key or token_b in key}
    if not params:
        raise RuntimeError(f"LoRA adapter {name!r} has no A/B parameters")
    if any(param.is_meta for param in params.values()):
        return None
    if _current_rank() != 0:
        return {}
    return {key: param.detach().cpu().clone() for key, param in params.items()}


def _restore_adapter(
    model: nn.Module,
    *,
    name: str,
    state_dict: dict[str, torch.Tensor],
) -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        from unirl.train.backend.sharded_state import load_model_state_dict

        load_model_state_dict(model, state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    if _current_rank() == 0:
        logger.info(
            "_restore_adapter(%r): restored %d deterministic tensor(s)",
            name,
            len(state_dict),
        )


@contextmanager
def _adapter_init_seed(seed: int | None) -> Iterator[None]:
    if seed is None:
        yield
        return

    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(int(seed))
        yield


@contextmanager
def adapters_disabled(model: nn.Module) -> Iterator[None]:
    """Temporarily route every PEFT LoRA layer through its frozen base weights.

    This mirrors PEFT's adapter-disabling behavior without changing
    ``requires_grad``. The beta KL reference replay wraps this in ``no_grad`` so
    the shared FSDP model can act as pi_ref while preserving the trainable adapter
    state.
    """
    from peft.tuners.lora import LoraLayer

    layers = [m for m in model.modules() if isinstance(m, LoraLayer)]
    prev = [bool(getattr(m, "_disable_adapters", False)) for m in layers]
    try:
        for m in layers:
            m._disable_adapters = True
        yield
    finally:
        for m, was_disabled in zip(layers, prev):
            m._disable_adapters = was_disabled


def _current_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


__all__ = ["adapters_disabled", "inject_lora"]
