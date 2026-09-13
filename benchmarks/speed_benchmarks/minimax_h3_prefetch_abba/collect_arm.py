#!/usr/bin/env python3
"""Collect resolved config, W&B metrics, and checksummed runtime artifacts for one arm."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from harness_lib import GateError, load_json, sha256_file, write_json_atomic


def select_metrics(wandb: dict) -> dict:
    if wandb.get("match_count") != 1:
        raise GateError("W&B extraction did not find exactly one named run")
    history = wandb["matches"][0].get("history") or []
    perf_rows = [row for row in history if "perf/step_time_s" in row]
    grad_rows = [row for row in history if "train/grad_norm" in row]
    reward_rows = [row for row in history if "rollout/reward_mean" in row]
    if len(perf_rows) != 1 or len(reward_rows) != 1:
        raise GateError("W&B history lacks exactly one perf and reward row")
    metrics = {
        key: perf_rows[0].get(key)
        for key in ("perf/step_time_s", "perf/generate_time_s", "perf/reward_time_s", "perf/train_time_s")
    }
    metrics["train/grad_norm"] = [row.get("train/grad_norm") for row in grad_rows]
    metrics["train/optimizer_updates"] = len(grad_rows)
    metrics["rollout/reward_mean"] = reward_rows[0].get("rollout/reward_mean")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    args = parser.parse_args()
    args.artifact.mkdir(parents=True, exist_ok=True)
    names = [
        f"{args.run_name}.log",
        f"{args.run_name}.status",
        f"{args.run_name}.started",
        f"{args.run_name}.identity",
        f"{args.run_name}.contract.json",
        f"{args.run_name}.receipt.json",
        f"sample_ledger_{args.run_name}.jsonl",
        f"update_membership_{args.run_name}.json",
        f"frozen_checkpoint_loaded_{args.run_name}.json",
        f"runtime_summary_{args.run_name}.json",
        f"nvml_{args.run_name}_{args.node}.csv",
        f"nvml_{args.run_name}_{args.node}.stderr",
    ]
    copied = {}
    for name in names:
        source = args.remote_root / name
        if not source.is_file():
            raise SystemExit(f"missing required arm artifact: {source}")
        target = args.artifact / name
        if target.exists():
            raise SystemExit(f"refusing existing artifact: {target}")
        shutil.copy2(source, target)
        copied[name] = sha256_file(target)

    hydra_source = args.remote_root / "hydra" / args.run_name / ".hydra" / "config.yaml"
    hydra_target = args.artifact / f"hydra_{args.run_name}_config.yaml"
    if not hydra_source.is_file() or hydra_target.exists():
        raise SystemExit("missing or existing resolved Hydra config")
    shutil.copy2(hydra_source, hydra_target)
    copied[hydra_target.name] = sha256_file(hydra_target)

    wandb_output = args.artifact / f"wandb_{args.run_name}.json"
    subprocess.run(
        [
            sys.executable,
            str(args.extractor),
            "--root",
            str(args.remote_root / "wandb" / args.run_name),
            "--run-name",
            args.run_name,
            "--output",
            str(wandb_output),
        ],
        check=True,
    )
    copied[wandb_output.name] = sha256_file(wandb_output)
    metrics = select_metrics(load_json(wandb_output))
    write_json_atomic(args.artifact / f"metrics_{args.run_name}.json", metrics)
    copied[f"metrics_{args.run_name}.json"] = sha256_file(args.artifact / f"metrics_{args.run_name}.json")
    write_json_atomic(
        args.artifact / f"collected_{args.run_name}.json",
        {"schema": "unirl-minimax-h3-p2-prefetch-collected-arm-v1", "run_name": args.run_name, "files": copied},
    )


if __name__ == "__main__":
    main()
