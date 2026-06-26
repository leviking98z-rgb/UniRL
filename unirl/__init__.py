"""Canonical Python package for this repository (`unirl`)."""

from __future__ import annotations

import importlib
import os
from typing import Dict, Tuple

__version__ = "0.1.0"


def _maybe_disable_cudnn() -> None:
    """Globally disable cuDNN + cuDNN-SDPA when requested (opt-in env gate).

    Workaround for the cuda-compat-13 + driver-535 stack: the cuDNN SDPA/conv
    backend aborts the process (SIGABRT / native crash, no Python exc) inside
    CLIP attention/conv — hit in the PickScore / CLIP reward encoders and any
    diffusion text-encoder, all running inside unirl Ray actors. Every actor
    runs ``import unirl``, so disabling here routes SDPA/conv through PyTorch's
    native CUDA kernels process-wide. Opt-in via ``UNIRL_DISABLE_CUDNN=1``
    (legacy ``DIFFRL_DISABLE_CUDNN=1`` also honored). Mirrors unirl_video.
    """
    if os.environ.get("UNIRL_DISABLE_CUDNN") != "1" and os.environ.get("DIFFRL_DISABLE_CUDNN") != "1":
        return
    try:
        import torch

        torch.backends.cudnn.enabled = False
        try:
            torch.backends.cuda.enable_cudnn_sdp(False)
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001 — best-effort; torch may be absent in CPU-only imports
        pass


_maybe_disable_cudnn()

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    # shared types
    "RewardRequest": ("unirl.types", "RewardRequest"),
    "RewardResponse": ("unirl.types", "RewardResponse"),
    "RewardType": ("unirl.types", "RewardType"),
    "SamplingRequirements": ("unirl.types.sampling", "SamplingRequirements"),
    # sde
    "get_sigma_schedule": ("unirl.sde", "get_sigma_schedule"),
    # reward
    "RewardBackend": ("unirl.reward.base", "RewardBackend"),
    # utils
    "load_function": ("unirl.utils", "load_function"),
    "set_seed": ("unirl.utils", "set_seed"),
    "configure_logger": ("unirl.utils", "configure_logger"),
}

__all__ = ["__version__", *_LAZY_ATTRS.keys()]


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
