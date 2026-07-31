"""Default rollout memory lifecycle independent of generation semantics."""

from __future__ import annotations

from typing import Dict

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed


class DefaultRolloutMemoryLifecycle:
    """No-op offload lifecycle with shared health and memory diagnostics.

    Engines with real offload behavior override ``sleep`` / ``wake_up`` and
    re-apply ``@distributed`` because Handle binds the most-derived attribute.
    """

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self) -> None:
        """Best-effort runtime offload. Default no-op."""

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(self) -> None:
        """Restore runtime resources after ``sleep``. Default no-op."""

    def onload_weights(self, *, track_prefix: str = "") -> None:
        """Restore the resources needed to receive a weight update."""
        del track_prefix
        self.wake_up()

    @property
    def is_offloaded(self) -> bool:
        """Whether the engine has released its runtime resources."""
        return False

    def health_check(self) -> bool:
        """Return True iff the engine is ready to serve generation."""
        return True

    def get_memory_info(self) -> Dict[str, float]:
        """Per-engine GPU allocator snapshot."""
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "cached_gb": torch.cuda.memory_reserved() / 1e9,
        }


__all__ = ["DefaultRolloutMemoryLifecycle"]
