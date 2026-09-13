#!/usr/bin/env python3
"""Extract one named W&B offline run into a deterministic JSON artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


def inspect(path: Path) -> dict[str, Any]:
    store = DataStore()
    store.open_for_scan(str(path))
    run: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    while True:
        data = store.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        kind = record.WhichOneof("record_type")
        if kind == "run":
            run = {"run_id": record.run.run_id, "display_name": record.run.display_name, "project": record.run.project}
        elif kind == "history":
            row: dict[str, Any] = {}
            for item in record.history.item:
                key = "/".join(item.nested_key)
                try:
                    row[key] = json.loads(item.value_json)
                except (json.JSONDecodeError, TypeError):
                    row[key] = item.value_json
            history.append(row)
    return {"path": str(path), "run": run, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matches = []
    for path in sorted(args.root.glob("wandb/offline-run-*/run-*.wandb")):
        result = inspect(path)
        if result["run"].get("display_name") == args.run_name:
            matches.append(result)
    payload = {"run_name": args.run_name, "root": str(args.root), "match_count": len(matches), "matches": matches}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one W&B run named {args.run_name!r}, found {len(matches)}")


if __name__ == "__main__":
    main()
