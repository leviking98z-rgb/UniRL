"""Process-local reproducibility and device-memory lifecycle helpers."""

from __future__ import annotations

import gc
import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int]) -> None:
    """Seed Python, NumPy, and PyTorch consistently for one process."""
    if seed is None:
        seed = int.from_bytes(os.urandom(8), "big") & 0x7FFFFFFF
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def clear_memory() -> None:
    """Synchronize and clear the local PyTorch CUDA cache."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()


__all__ = ["clear_memory", "set_seed"]
