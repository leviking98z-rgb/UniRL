"""Local, structured experiment records for framework performance comparisons."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

SCHEMA_VERSION = 1


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


class ExperimentRecorder:
    """Append one run header and per-step metrics to a JSONL file.

    The recorder is deliberately independent of WandB. This makes the same
    artifact available in offline clusters, CI, and local smoke tests.
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
