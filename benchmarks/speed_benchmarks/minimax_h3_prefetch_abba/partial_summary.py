#!/usr/bin/env python3
"""Write an atomic, explicitly non-result partial summary after harness failure."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from harness_lib import write_json_atomic


def build_partial_summary(
    artifact: Path,
    *,
    campaign_id: str,
    exit_code: int,
    failed_line: int,
    failed_command: str,
    current_run: str | None,
) -> dict[str, Any]:
    artifact = artifact.resolve()
    available = (
        sorted(path.relative_to(artifact).as_posix() for path in artifact.rglob("*") if path.is_file())
        if artifact.is_dir()
        else []
    )
    return {
        "schema": "unirl-minimax-h3-p2-prefetch-abba-partial-summary-v1",
        "campaign_id": campaign_id,
        "completed": False,
        "performance_usable": False,
        "exit_code": int(exit_code),
        "failed_line": int(failed_line),
        "failed_command": failed_command,
        "current_run": current_run or None,
        "available_files": available,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_partial_summary(
    artifact: Path,
    *,
    campaign_id: str,
    exit_code: int,
    failed_line: int,
    failed_command: str,
    current_run: str | None,
) -> dict[str, Any]:
    artifact.mkdir(parents=True, exist_ok=True)
    output = artifact / "partial-summary.json"
    payload = build_partial_summary(
        artifact,
        campaign_id=campaign_id,
        exit_code=exit_code,
        failed_line=failed_line,
        failed_command=failed_command,
        current_run=current_run,
    )
    write_json_atomic(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--failed-line", type=int, required=True)
    parser.add_argument("--failed-command", required=True)
    parser.add_argument("--current-run")
    args = parser.parse_args()
    payload = write_partial_summary(
        args.artifact,
        campaign_id=args.campaign_id,
        exit_code=args.exit_code,
        failed_line=args.failed_line,
        failed_command=args.failed_command,
        current_run=args.current_run,
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
