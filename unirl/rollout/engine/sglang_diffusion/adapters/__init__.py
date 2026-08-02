"""Lazy model-adapter registry for the ``sglang_diffusion`` engine.

Importing this package declares adapter dotpaths without importing concrete
implementations. :func:`get_adapter` imports only the module selected by the
active ``model_family``. Historical class exports remain available through
module-level lazy attribute resolution.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from unirl.rollout.engine.sglang_diffusion.adapters.base import (
    ModelAdapter,
    get_adapter,
    register_adapter,
    register_lazy_adapter,
    registered_adapters,
)
from unirl.rollout.engine.sglang_diffusion.adapters.image import ImageAdapter

_ADAPTER_SPECS = {
    "sd3": "unirl.rollout.engine.sglang_diffusion.adapters.sd3:SD3Adapter",
    "flux": "unirl.rollout.engine.sglang_diffusion.adapters.flux:FluxAdapter",
    "flux2_klein": "unirl.rollout.engine.sglang_diffusion.adapters.flux:Flux2KleinAdapter",
    "qwen_image": "unirl.rollout.engine.sglang_diffusion.adapters.qwen_image:QwenImageAdapter",
    "qwen_image_edit_plus": (
        "unirl.rollout.engine.sglang_diffusion.adapters.qwen_image_edit_plus:QwenImageEditPlusAdapter"
    ),
    "z_image": "unirl.rollout.engine.sglang_diffusion.adapters.z_image:ZImageAdapter",
    "mochi": "unirl.rollout.engine.sglang_diffusion.adapters.video.base:MochiAdapter",
    "hunyuan_video": "unirl.rollout.engine.sglang_diffusion.adapters.video.hunyuan:HunyuanVideoAdapter",
    "wan21": "unirl.rollout.engine.sglang_diffusion.adapters.video.wan:Wan21T2VAdapter",
    "wan22": "unirl.rollout.engine.sglang_diffusion.adapters.video.wan:Wan22T2VAdapter",
    "ltx2": "unirl.rollout.engine.sglang_diffusion.adapters.video.ltx:Ltx2T2VAdapter",
}

for _family, _dotpath in _ADAPTER_SPECS.items():
    register_lazy_adapter(_family, _dotpath)

_CLASS_TO_FAMILY = {dotpath.rpartition(":")[2]: family for family, dotpath in _ADAPTER_SPECS.items()}
_BASE_EXPORTS = {
    "VideoAdapter": "unirl.rollout.engine.sglang_diffusion.adapters.video.base:VideoAdapter",
}


def __getattr__(name: str) -> Any:
    family = _CLASS_TO_FAMILY.get(name)
    if family is not None:
        value = get_adapter(family)
    else:
        dotpath = _BASE_EXPORTS.get(name)
        if dotpath is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        module_name, _, attribute = dotpath.partition(":")
        value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_CLASS_TO_FAMILY) | set(_BASE_EXPORTS))


__all__ = [
    "ModelAdapter",
    "ImageAdapter",
    "VideoAdapter",
    "get_adapter",
    "register_adapter",
    "register_lazy_adapter",
    "registered_adapters",
    "SD3Adapter",
    "FluxAdapter",
    "Flux2KleinAdapter",
    "QwenImageAdapter",
    "QwenImageEditPlusAdapter",
    "Wan22T2VAdapter",
    "Wan21T2VAdapter",
    "Ltx2T2VAdapter",
    "MochiAdapter",
    "HunyuanVideoAdapter",
    "ZImageAdapter",
]
