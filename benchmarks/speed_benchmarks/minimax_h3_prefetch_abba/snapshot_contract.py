#!/usr/bin/env python3
"""Materialize a self-contained, node-visible snapshot of the P2 integration contract."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any

from harness_lib import GateError, load_json, normalize_path, sha256_file, validate_p0_contract, write_json_atomic

_FILE_BINDINGS = (
    (("source", "archive"), "source-archive"),
    (("source", "tree_manifest"), "source-tree"),
    (("source", "p1_source_preflight"), "p1-source-preflight"),
    (("p0_design",), "p0-design"),
    (("prompt_manifest",), "prompts"),
    (("frozen_checkpoint", "initial_lora_audit"), "frozen-initial-lora-audit"),
    (("runtime_overlay", "module"), "p0-runtime-overlay"),
)


def _at(payload: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    value: Any = payload
    for component in path:
        if not isinstance(value, dict) or component not in value:
            raise GateError(f"integration contract lacks {'.'.join(path)}")
        value = value[component]
    if not isinstance(value, dict):
        raise GateError(f"integration contract {'.'.join(path)} is not an object")
    return value


def snapshot_contract(
    source_path: Path,
    output_path: Path,
    input_dir: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Copy every file binding and rewrite only paths, preserving all identities."""
    source_path = source_path.resolve()
    validated_source = validate_p0_contract(source_path)
    payload = copy.deepcopy(load_json(source_path))
    input_dir = input_dir.resolve()
    output_path = output_path.resolve()
    receipt_path = receipt_path.resolve()
    if input_dir.exists() or output_path.exists() or receipt_path.exists():
        raise GateError("refusing existing contract snapshot output")
    input_dir.mkdir(parents=True)

    copied: dict[str, dict[str, Any]] = {}
    try:
        for key_path, stem in _FILE_BINDINGS:
            binding = _at(payload, key_path)
            raw_path = binding.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                raise GateError(f"{'.'.join(key_path)}.path is required")
            source = normalize_path(raw_path, base=source_path.parent)
            suffixes = "".join(source.suffixes)
            destination = input_dir / f"{stem}{suffixes}"
            shutil.copy2(source, destination)
            observed = sha256_file(destination)
            if observed != binding.get("sha256"):
                raise GateError(f"snapshot digest mismatch for {'.'.join(key_path)}")
            binding["path"] = os.fspath(destination)
            copied[".".join(key_path)] = {
                "path": os.fspath(destination),
                "sha256": observed,
                "bytes": destination.stat().st_size,
            }

        frozen = _at(payload, ("frozen_checkpoint",))
        load_dir = frozen.get("load_dir")
        if not isinstance(load_dir, str) or not load_dir:
            raise GateError("frozen_checkpoint.load_dir is required")
        frozen["load_dir"] = os.fspath(normalize_path(load_dir, base=source_path.parent))

        write_json_atomic(output_path, payload)
        validated_snapshot = validate_p0_contract(output_path)
        receipt = {
            "schema": "unirl-minimax-h3-p2-contract-snapshot-v1",
            "completed": True,
            "source_contract": {
                "path": os.fspath(source_path),
                "sha256": validated_source["sha256"],
            },
            "snapshot_contract": {
                "path": os.fspath(output_path),
                "sha256": validated_snapshot["sha256"],
            },
            "copied_bindings": copied,
        }
        write_json_atomic(receipt_path, receipt)
        return receipt
    except BaseException:
        shutil.rmtree(input_dir, ignore_errors=True)
        output_path.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = snapshot_contract(args.source, args.output, args.input_dir, args.receipt)
    except GateError as exc:
        print(json.dumps({"completed": False, "error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
