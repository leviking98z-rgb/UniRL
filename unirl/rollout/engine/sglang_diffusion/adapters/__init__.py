"""Model adapters for the ``sglang_diffusion`` engine.

Concrete adapter modules are imported on first use. This keeps an unrelated
adapter's optional dependencies and import failures out of the selected model's
startup path while preserving the package's public class exports.
"""

from importlib import import_module

from unirl.rollout.engine.sglang_diffusion.adapters.base import (
    _BUILTIN_ADAPTERS,
    ModelAdapter,
    get_adapter,
    register_adapter,
    registered_adapters,
)
from unirl.rollout.engine.sglang_diffusion.adapters.image import ImageAdapter

_CLASS_MODULES = {
    class_name: module_name
    for module_name, class_name in (target.split(":", 1) for target in _BUILTIN_ADAPTERS.values())
}
_CLASS_MODULES["VideoAdapter"] = "unirl.rollout.engine.sglang_diffusion.adapters.video"


def __getattr__(name: str):
    module_name = _CLASS_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "ModelAdapter",
    "ImageAdapter",
    "get_adapter",
    "register_adapter",
    "registered_adapters",
    "SD3Adapter",
    "FluxAdapter",
    "Flux2KleinAdapter",
    "QwenImageAdapter",
    "VideoAdapter",
    "QwenImageEditPlusAdapter",
    "Wan22T2VAdapter",
    "Wan21T2VAdapter",
    "MochiAdapter",
    "HunyuanVideoAdapter",
    "ZImageAdapter",
]
