"""Local, structured experiment records for framework performance comparisons."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

SCHEMA_VERSION = 2


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _git_state() -> Dict[str, Any]:
    """Commit + dirty flag for the tree this process is running from.

    A speed comparison is only meaningful between runs of KNOWN code. Recording
    the commit lets the comparison tool refuse arms that silently differ — the
    failure mode where a "candidate" is measured against a baseline built from
    another revision, and the delta is attributed to the change under test.

    Resolution order, cheapest first:

    1. ``UNIRL_EXPERIMENT_COMMIT`` / ``UNIRL_EXPERIMENT_DIRTY`` if the launcher
       already knows (it usually does — it just checked out the revision). This
       is the intended path for cluster runs.
    2. Shelling out to ``git``, but only when ``UNIRL_EXPERIMENT_GIT_PROBE=1``.
       Off by default: on this cluster's CephFS work dirs even ``git rev-parse``
       takes 6-10s and ``git status`` exceeds 30s, and a telemetry field must
       never delay a training launch.

    ``dirty`` is None when unverified. None must NOT be read as "clean" — the
    consumer in :mod:`benchmarks.framework.effect_size` treats it as unverified
    and blocks acceptance unless overridden.
    """
    env_commit = os.environ.get("UNIRL_EXPERIMENT_COMMIT")
    env_dirty = os.environ.get("UNIRL_EXPERIMENT_DIRTY")
    state: Dict[str, Any] = {
        "commit": env_commit or None,
        "dirty": None if env_dirty is None else env_dirty.strip().lower() in ("1", "true", "yes"),
        "branch": os.environ.get("UNIRL_EXPERIMENT_BRANCH") or None,
        "source": "env" if env_commit else None,
    }
    if state["commit"] is not None or os.environ.get("UNIRL_EXPERIMENT_GIT_PROBE") != "1":
        return state

    root = Path(__file__).resolve().parents[2]

    def _git(*args: str, timeout: float) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = _git("rev-parse", "HEAD", timeout=20.0)
    # Tracked files only: untracked scratch in a work dir is not a code
    # difference, and skipping it avoids the slowest part of the tree walk.
    status = _git("status", "--porcelain", "--untracked-files=no", timeout=60.0)
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status.strip()),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", timeout=20.0),
        "source": "git-probe",
    }


def _environment_state() -> Dict[str, Any]:
    """Host / accelerator identity — the other half of a comparable arm.

    Two arms measured on different nodes are not a controlled A/B: this cluster
    mixes GPU counts and driver versions, and step time tracks both.
    """
    env: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["cuda"] = torch.version.cuda
            env["device_count"] = int(torch.cuda.device_count())
            env["device_name"] = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover - torch absent in control-plane use
        pass
    return env


class ExperimentRecorder:
    """Append one run header and per-step metrics to a JSONL file.

    The recorder is deliberately independent of WandB. This makes the same
    artifact available in offline clusters, CI, and local smoke tests.

    The ``run_start`` header carries ``git`` and ``environment`` blocks so a
    downstream comparison can verify the two arms differ ONLY in the knob under
    test (see :mod:`benchmarks.framework.effect_size`).
    """

    def __init__(
        self,
        output: Optional[str],
        *,
        run_name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        rank: int = 0,
        append: bool = False,
    ) -> None:
        self.path = Path(output).expanduser().resolve() if output and rank == 0 else None
        self._file = None
        if self.path is None:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a" if append else "x", encoding="utf-8")
        self._write(
            {
                "record_type": "run_start",
                "run_name": run_name,
                "pid": os.getpid(),
                "git": _git_state(),
                "environment": _environment_state(),
                "metadata": _json_safe(metadata or {}),
            }
        )

    @property
    def enabled(self) -> bool:
        return self._file is not None

    def _write(self, payload: Dict[str, Any]) -> None:
        if self._file is None:
            return
        record = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        self._file.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self._file.flush()

    def log_step(self, step: int, metrics: Mapping[str, Any]) -> None:
        self._write(
            {
                "record_type": "step",
                "step": int(step),
                "metrics": _json_safe(metrics),
            }
        )

    def finish(self) -> None:
        if self._file is None:
            return
        self._write({"record_type": "run_end"})
        self._file.close()
        self._file = None
