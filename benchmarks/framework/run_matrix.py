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


def resolve_provenance() -> dict[str, str]:
    """Resolve the repo revision ONCE, here, for all runs in this matrix.

    The training process cannot afford to shell out to git — on this cluster's
    network work dirs even ``git rev-parse`` costs 6-10s and ``git status``
    exceeds 30s. The launcher pays that cost a single time and passes the answer
    down through the environment, so every record self-describes without any
    per-run startup penalty. Values already in the environment win, which lets an
    outer orchestrator that knows the revision skip the probe entirely.
    """
    resolved: dict[str, str] = {}
    if os.environ.get("UNIRL_EXPERIMENT_COMMIT"):
        return resolved
    root = Path(__file__).resolve().parents[2]

    def _git(*args: str, timeout: float) -> str | None:
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

    commit = _git("rev-parse", "HEAD", timeout=30.0)
    if commit:
        resolved["UNIRL_EXPERIMENT_COMMIT"] = commit
    branch = _git("rev-parse", "--abbrev-ref", "HEAD", timeout=30.0)
    if branch:
        resolved["UNIRL_EXPERIMENT_BRANCH"] = branch
    status = _git("status", "--porcelain", "--untracked-files=no", timeout=120.0)
    # Only assert cleanliness when the check actually completed; leaving the var
    # unset records dirty=None, which the effect gate treats as unverified.
    if status is not None:
        resolved["UNIRL_EXPERIMENT_DIRTY"] = "1" if status.strip() else "0"
    return resolved


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
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help=(
            "repeat each workload N times into replicate-<i>/ subdirectories. "
            "N>=3 is required by the effect-size gate: a single run cannot "
            "separate a real speedup from this cluster's run-to-run spread."
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="extra Hydra override appended to every selected workload (repeatable)",
    )
    args = parser.parse_args()
    if args.replicates < 1:
        raise SystemExit("--replicates must be >= 1")

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

    # Resolved once for the whole matrix, before any run starts.
    provenance = resolve_provenance()
    if provenance:
        print(f"provenance: {provenance}")

    for name, spec in selected:
        effective = dict(spec)
        if args.override:
            effective["overrides"] = [*spec.get("overrides", []), *args.override]
        command = build_command(effective)
        print(f"{name}: {' '.join(command)}" + (f"  x{args.replicates}" if args.replicates > 1 else ""))
        if not args.execute:
            continue

        for replicate in range(args.replicates):
            # Replicates are separate directories, never separate steps in one
            # file: steps inside a run share caches, allocator state and a
            # compile, so they are not independent samples of the config.
            run_dir = args.output_dir / name
            if args.replicates > 1:
                run_dir = run_dir / f"replicate-{replicate}"
            run_dir = run_dir.resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            experiment_path = run_dir / "experiment.jsonl"
            if experiment_path.exists():
                raise SystemExit(f"refusing to mix runs: {experiment_path} already exists")
            env = os.environ.copy()
            env["UNIRL_EXPERIMENT_OUTPUT"] = str(experiment_path)
            env.update(provenance)
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "name": name,
                        "replicate": replicate,
                        "spec": effective,
                        "command": command,
                        "provenance": provenance,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            with (run_dir / "train.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            if completed.returncode:
                raise SystemExit(
                    f"{name} replicate {replicate} failed with exit code {completed.returncode}; "
                    f"see {run_dir / 'train.log'}"
                )

            health_command = build_health_command(effective, experiment_path, run_dir / "health.json")
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
                    f"{name} replicate {replicate} completed but failed its health gate; see {run_dir / 'health.log'}"
                )


if __name__ == "__main__":
    main()
