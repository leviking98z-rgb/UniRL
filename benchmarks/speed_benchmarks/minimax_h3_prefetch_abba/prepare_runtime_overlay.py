#!/usr/bin/env python3
"""Prepare the immutable node-local P2 runtime overlay from bound inputs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from harness_lib import GateError, load_json, sha256_file, validate_p0_contract, write_json_atomic


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> str:
    if not source.is_file() or sha256_file(source) != expected_sha256:
        raise GateError(f"runtime overlay source identity mismatch: {source}")
    shutil.copy2(source, destination)
    observed = sha256_file(destination)
    if observed != expected_sha256:
        raise GateError(f"runtime overlay copy identity mismatch: {destination}")
    return observed


def prepare(contract_path: Path, plan_path: Path, destination: Path) -> dict:
    contract = validate_p0_contract(contract_path.resolve())
    plan = load_json(plan_path.resolve())
    overlay = plan.get("runtime_overlay")
    if not isinstance(overlay, dict):
        raise GateError("plan lacks runtime_overlay")
    checks = {
        "event_schema": overlay.get("event_schema") == contract["runtime_overlay"]["event_schema"],
        "base_path": overlay.get("base_module_path") == contract["runtime_overlay"]["module_binding"]["path"],
        "base_sha": overlay.get("base_module_sha256") == contract["runtime_overlay"]["module_binding"]["sha256"],
    }
    if not all(checks.values()):
        raise GateError(f"plan/runtime-overlay contract mismatch: {checks}")
    if destination.exists():
        raise GateError(f"refusing existing runtime overlay directory: {destination}")
    destination.mkdir(parents=True)
    try:
        copied = {
            "p0_runtime_receipts_base.py": _copy_verified(
                Path(overlay["base_module_path"]),
                destination / "p0_runtime_receipts_base.py",
                overlay["base_module_sha256"],
            ),
            "p2_runtime_receipts.py": _copy_verified(
                Path(overlay["adapter_module_path"]),
                destination / "p2_runtime_receipts.py",
                overlay["adapter_module_sha256"],
            ),
            "sitecustomize.py": _copy_verified(
                Path(overlay["sitecustomize_path"]),
                destination / "sitecustomize.py",
                overlay["sitecustomize_sha256"],
            ),
        }
        payload = {
            "schema": "unirl-minimax-h3-p2-runtime-overlay-bundle-v1",
            "completed": True,
            "event_schema": overlay["event_schema"],
            "p0_contract_sha256": contract["sha256"],
            "files": copied,
            "root": str(destination.resolve()),
        }
        write_json_atomic(destination / "manifest.json", payload)
        return payload
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0-contract", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = prepare(
            args.p0_contract,
            args.plan,
            args.destination.resolve(),
        )
    except GateError as exc:
        payload = {
            "schema": "unirl-minimax-h3-p2-runtime-overlay-bundle-v1",
            "completed": False,
            "error": str(exc),
        }
        write_json_atomic(args.output, payload)
        print(json.dumps(payload, sort_keys=True))
        raise SystemExit(2) from exc
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
