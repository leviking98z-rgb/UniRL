"""Opt-in bootstrap for the P2-compatible P0 runtime receipt adapter."""

from __future__ import annotations

import os
import sys


def _is_target_process() -> bool:
    """Limit monkeypatching to the UniRL driver and its Ray workers."""
    if os.environ.get("P0_AUDIT_ENABLE") != "1":
        return False
    if os.environ.get("P0_AUDIT_FORCE_INSTALL") == "1":
        return True
    argv = " ".join(sys.argv)
    executable = os.path.basename(sys.argv[0] if sys.argv else "")
    if "unirl.train_diffusion" in argv:
        return True
    return bool(
        os.environ.get("RAY_RAYLET_PID")
        or os.environ.get("RAY_JOB_ID")
        or os.environ.get("RAY_WORKER_ID")
        or executable in {"default_worker.py", "ray::IDLE"}
    )


if _is_target_process():
    from p2_runtime_receipts import install

    install()
