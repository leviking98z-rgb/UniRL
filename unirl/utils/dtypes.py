"""Compatibility facade for :mod:`unirl.config.dtypes`."""

from unirl.config.dtypes import canonical_torch_dtype_name, inject_model_dtype_kwarg, parse_torch_dtype

__all__ = [
    "canonical_torch_dtype_name",
    "inject_model_dtype_kwarg",
    "parse_torch_dtype",
]
