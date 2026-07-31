"""Compatibility facade for :mod:`unirl.rollout.endpoint`."""

from unirl.rollout.endpoint import (
    format_scheduler_endpoint,
    normalize_scheduler_host,
    parse_scheduler_endpoint,
    parse_scheduler_endpoint_pool,
)

__all__ = [
    "format_scheduler_endpoint",
    "normalize_scheduler_host",
    "parse_scheduler_endpoint",
    "parse_scheduler_endpoint_pool",
]
