"""Narrow compatibility patches for torch-native FSDP2."""

from __future__ import annotations

import hashlib
import inspect

import torch

_TORCH_27_RELEASES = {(2, 7, 0), (2, 7, 1)}
_TORCH_27_INIT_MP_DTYPES_SHA256 = "45148e36aa568c13b423a7ebcc175f3232098eff4f9aed99b9f6adcda9a5e379"


def install_torch27_mixed_dtype_no_grad_compat() -> bool:
    """Backport PyTorch #154103 for the exact affected 2.7 FSDP2 surface; see train/readme.md."""
    raw_release = torch.__version__.split("+", 1)[0].split(".")[:3]
    try:
        release = tuple(int(part) for part in raw_release)
    except ValueError as exc:
        raise RuntimeError(f"cannot parse torch version {torch.__version__!r}") from exc
    if release not in _TORCH_27_RELEASES:
        return False

    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

    current = FSDPParamGroup._init_mp_dtypes
    if getattr(current, "_unirl_torch27_mixed_dtype_no_grad", False):
        return True
    try:
        source = inspect.getsource(current)
    except (OSError, TypeError) as exc:
        raise RuntimeError("cannot inspect torch 2.7 FSDPParamGroup._init_mp_dtypes") from exc
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != _TORCH_27_INIT_MP_DTYPES_SHA256:
        raise RuntimeError(
            "torch 2.7 FSDPParamGroup._init_mp_dtypes drifted from the supported 2.7.0/2.7.1 surface: "
            f"expected sha256={_TORCH_27_INIT_MP_DTYPES_SHA256}, got {digest}"
        )

    def _init_mp_dtypes(self) -> None:
        for fsdp_param in self.fsdp_params:
            fsdp_param.init_dtype_attrs(self.mp_policy)
        trainable_params = [p for p in self.fsdp_params if p.sharded_param.requires_grad]
        orig_dtypes = {p.orig_dtype for p in trainable_params}
        reduce_dtypes = {p.reduce_dtype for p in trainable_params}
        if len(trainable_params) > 0 and len(orig_dtypes) != 1:
            raise AssertionError(f"FSDP expects uniform original parameter dtype but got {orig_dtypes}")
        self._orig_dtype = next(iter(orig_dtypes)) if len(trainable_params) else None
        if len(trainable_params) > 0 and len(reduce_dtypes) != 1:
            raise AssertionError(f"FSDP expects uniform reduce dtype but got {reduce_dtypes}")
        self._reduce_dtype = next(iter(reduce_dtypes)) if len(trainable_params) else None

    _init_mp_dtypes._unirl_torch27_mixed_dtype_no_grad = True
    FSDPParamGroup._init_mp_dtypes = _init_mp_dtypes
    return True


__all__ = ["install_torch27_mixed_dtype_no_grad_compat"]
