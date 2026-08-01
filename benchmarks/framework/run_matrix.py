"""Expand or execute the framework performance workload matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


def build_command(spec: Mapping[str, Any]) -> list[str]:
    return [
        sys.executable,
        "-m",
        str(spec["entrypoint"]),
        f"--config-name={spec['config_name']}",
        *[str(value) for value in spec.get("overrides", [])],
    ]


def build_health_command(
    spec: Mapping[str, Any],
    experiment_path: Path,
    output_path: Path,
) -> list[str] | None:
    policy = spec.get("health_policy")
    if not policy:
        return None
    return [
        sys.executable,
        "-m",
        "benchmarks.framework.check_run",
        str(experiment_path),
        "--policy",
        str(policy),
        "--json-output",
        str(output_path),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(__file__).with_name("workloads.yaml"),
    )
    parser.add_argument("--workload", action="append", help="workload name; repeat to select several")
    parser.add_argument("--tag", help="select workloads containing this tag")
    parser.add_argument("--output-dir", type=Path, default=Path("framework_results"))
    parser.add_argument("--execute", action="store_true", help="execute sequentially; default is a dry-run")
    args = parser.parse_args()

    document = json.loads(args.matrix.read_text(encoding="utf-8"))
    workloads = document.get("workloads", {})
    selected = []
    for name, spec in workloads.items():
        if args.workload and name not in args.workload:
            continue
        if args.tag and args.tag not in spec.get("tags", []):
            continue
        selected.append((name, spec))
    if not selected:
        raise SystemExit("no workloads selected")

    for name, spec in selected:
        command = build_command(spec)
        print(f"{name}: {' '.join(command)}")
        if not args.execute:
            continue

        run_dir = (args.output_dir / name).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        experiment_path = run_dir / "experiment.jsonl"
        if experiment_path.exists():
            raise SystemExit(f"refusing to mix runs: {experiment_path} already exists")
        env = os.environ.copy()
        env["UNIRL_EXPERIMENT_OUTPUT"] = str(experiment_path)
        (run_dir / "manifest.json").write_text(
            json.dumps({"name": name, "spec": spec, "command": command}, indent=2) + "\n",
            encoding="utf-8",
        )
        with (run_dir / "train.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise SystemExit(f"{name} failed with exit code {completed.returncode}; see {run_dir / 'train.log'}")

        health_command = build_health_command(spec, experiment_path, run_dir / "health.json")
        if health_command is None:
            continue
        with (run_dir / "health.log").open("w", encoding="utf-8") as log:
            health = subprocess.run(
                health_command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if health.returncode:
            raise SystemExit(
                f"{name} completed but failed its health gate; see {run_dir / 'health.log'}"
            )


if __name__ == "__main__":
    main()
