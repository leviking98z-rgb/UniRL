"""Driver-side ``RolloutReq``↔``RolloutResp`` conversion: the adapter ABC + registry.

A thin top ABC (registry + boilerplate with sensible defaults) over a per-output-
shape base adapter (:mod:`image`) that holds the conversion logic as overridable
methods. Concrete adapters override only what differs and self-register by
``model_family`` key. Selected once at engine construction via :func:`get_adapter`.

Pure: never imports SGLang — adapters consume the seam's ``RawResult`` protocol
(a structural view of ``GenerationResult``), not the runtime. The adapter is
bound to the engine config + model config at construction so its conversion
methods don't thread them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple

from unirl.config.require import require
from unirl.rollout.engine.sglang_diffusion.backends import RawResult
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp

# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_REGISTRY: Dict[str, type["ModelAdapter"]] = {}
_BUILTIN_ADAPTERS = {
    "flux": "unirl.rollout.engine.sglang_diffusion.adapters.flux:FluxAdapter",
    "flux2_klein": "unirl.rollout.engine.sglang_diffusion.adapters.flux:Flux2KleinAdapter",
    "hunyuan_video": "unirl.rollout.engine.sglang_diffusion.adapters.video:HunyuanVideoAdapter",
    "mochi": "unirl.rollout.engine.sglang_diffusion.adapters.video:MochiAdapter",
    "qwen_image": "unirl.rollout.engine.sglang_diffusion.adapters.qwen_image:QwenImageAdapter",
    "qwen_image_edit_plus": (
        "unirl.rollout.engine.sglang_diffusion.adapters.qwen_image_edit_plus:QwenImageEditPlusAdapter"
    ),
    "sd3": "unirl.rollout.engine.sglang_diffusion.adapters.sd3:SD3Adapter",
    "wan21": "unirl.rollout.engine.sglang_diffusion.adapters.video:Wan21T2VAdapter",
    "wan22": "unirl.rollout.engine.sglang_diffusion.adapters.video:Wan22T2VAdapter",
    "z_image": "unirl.rollout.engine.sglang_diffusion.adapters.z_image:ZImageAdapter",
}


def register_adapter(key: str):
    """Class decorator: register an adapter under its ``model_family`` key."""

    def deco(cls: type["ModelAdapter"]) -> type["ModelAdapter"]:
        builtin = _BUILTIN_ADAPTERS.get(key)
        if builtin is not None:
            expected_module, expected_name = builtin.split(":", 1)
            require(
                (cls.__module__, cls.__name__) == (expected_module, expected_name),
                f"adapter key {key!r} is reserved for {builtin}",
            )
        require(
            key not in _REGISTRY,
            f"adapter key {key!r} already registered by {_REGISTRY.get(key)!r}",
        )
        _REGISTRY[key] = cls
        cls.model_family = key
        return cls

    return deco


def get_adapter(key: str) -> type["ModelAdapter"]:
    """Look up an adapter class, importing its module only on first use."""
    if key in _REGISTRY:
        return _REGISTRY[key]

    require(
        key in _BUILTIN_ADAPTERS,
        f"unknown model_family {key!r}; registered: {list(registered_adapters())}",
    )
    module_name, class_name = _BUILTIN_ADAPTERS[key].split(":", 1)
    module = import_module(module_name)
    adapter_cls = getattr(module, class_name)
    require(
        _REGISTRY.get(key) is adapter_cls,
        f"built-in adapter {key!r} did not register {_BUILTIN_ADAPTERS[key]}",
    )
    return adapter_cls


def registered_adapters() -> Tuple[str, ...]:
    return tuple(sorted(_BUILTIN_ADAPTERS.keys() | _REGISTRY.keys()))


# --------------------------------------------------------------------------- #
# ABC
# --------------------------------------------------------------------------- #


class ModelAdapter(ABC):
    """Thin ABC: registry key + boilerplate defaults + the two conversion seams.

    The conversion *logic* lives on the per-shape base adapter (``ImageAdapter``);
    this ABC only declares the boilerplate every adapter shares (SDE-label resolution,
    schedule policy, LoRA spec, server-boot extras, validation) and the two abstract
    methods the engine drives.
    """

    model_family: str = ""

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None) -> None:
        self.cfg = config
        self.model_config = model_config
        self._sde_label = self.resolve_sde_label(strategy)
        self.validate()

    # ---- SDE kernel label (boilerplate; overridable) ----
    @staticmethod
    def resolve_sde_label(strategy: Any) -> Optional[str]:
        """Map the SDE strategy to SGLang's ``rollout_sde_type`` kernel label.

        ``None`` when no strategy (ODE/eval/NFT — the SDE branch of
        ``build_inputs`` is then skipped). The SGLang fork must register a kernel
        under the returned string whose update math matches UniRL's bit-for-bit.
        """
        if strategy is None:
            return None
        canonical = type(strategy).canonical_name.strip().lower()
        if canonical == "flow":
            return "sde"
        if canonical == "cps":
            return "cps"
        if canonical == "dance":
            return "dance"
        raise ValueError(
            f"SGLang rollout supports sde_type in {{'flow', 'cps', 'dance'}} only "
            f"(each has a verified SGLang-side kernel matching UniRL's math); got "
            f"canonical={canonical!r}. Switch the SDE strategy or add a mapping "
            f"after verifying the SGLang kernel is mathematically equivalent."
        )

    # ---- model-specific ServerArgs extras (override hook; default none) ----
    def boot_kwargs(self) -> Dict[str, Any]:
        """Extra SGLang ServerArgs intent a model needs beyond the generic set.

        The generic server kwargs (model_path, parallelism, LoRA hints) are
        derived in ``config.server_intent``; override this only when a model
        needs an additional ServerArgs knob.
        """
        return {}

    # ---- schedule policy (default: generic FlowMatch) ----
    def schedule_policy(self) -> Any:
        """The σ schedule policy. Default reads ``model_config.shift`` (+ optional
        dynamic-shift hints); Klein-style models override with a factory."""
        from unirl.sde.runtime import FlowMatchSchedulePolicy

        mc = self.model_config
        return FlowMatchSchedulePolicy.from_pretrained(
            mc.pretrained_model_ckpt_path,
            shift=float(mc.shift),
            require_dynamic=bool(getattr(mc, "use_dynamic_shifting", False)),
            dynamic_overrides=getattr(mc, "dynamic_shift_overrides", None),
        )

    # ---- LoRA spec: (pipeline_prefix, target_modules) ----
    def lora_spec(self) -> Tuple[str, List[str]]:
        prefix = str(self.model_config.weight_sync_param_name_prefix or "")
        target_modules = list(self.cfg.target_modules or ("transformer",))
        return prefix, target_modules

    # ---- validation (default: require shift + ckpt) ----
    def validate(self) -> None:
        mc = self.model_config
        require(
            mc is not None and bool(getattr(mc, "pretrained_model_ckpt_path", None)),
            f"{type(self).__name__} requires model_config.pretrained_model_ckpt_path",
        )
        require(
            hasattr(mc, "shift"),
            f"{type(self).__name__} requires model_config.shift; got "
            f"{type(mc).__name__}. Use a registered model preset.",
        )

    # ---- the two conversion seams the engine drives ----
    @abstractmethod
    def build_inputs(self, req: RolloutReq, *, initial_noise: Any) -> Dict[str, Any]:
        """Translate a ``RolloutReq`` into SGLang ``generate`` sampling kwargs."""

    @abstractmethod
    def build_response(self, req: RolloutReq, raw: List[RawResult]) -> RolloutResp:
        """Translate SGLang's results back into a typed ``RolloutResp``."""


__all__ = [
    "ModelAdapter",
    "register_adapter",
    "get_adapter",
    "registered_adapters",
]
