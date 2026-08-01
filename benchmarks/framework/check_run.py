"""Validate one UniRL experiment record against absolute health gates."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from benchmarks.framework.compare_runs import reduce_metric


@dataclass(frozen=True)
class RunMetricResult:
    metric: str
    value: float | None
    passed: bool
    reason: str


def load_run(path: Path) -> tuple[list[dict[str, float]], bool]:
    steps: list[dict[str, float]] = []
    record_types: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            record_type = record.get("record_type")
            if isinstance(record_type, str):
                record_types.append(record_type)
            if record_type != "step":
                continue
            metrics = record.get("metrics")
            if isinstance(metrics, dict):
                steps.append(
                    {
                        str(key): float(value)
                        for key, value in metrics.items()
                        if isinstance(value, (int, float)) and math.isfinite(float(value))
                    }
                )
    complete = bool(record_types) and record_types[0] == "run_start" and record_types[-1] == "run_end"
    return steps, complete


def check_metric(metric: str, value: float | None, rule: Mapping[str, Any]) -> RunMetricResult:
    required = bool(rule.get("required", True))
    if value is None:
        return RunMetricResult(metric, None, not required, "missing metric")

    failures: list[str] = []
    min_value = rule.get("min_value")
    if min_value is not None and value < float(min_value):
        failures.append(f"{value:.6g} < {float(min_value):.6g}")
    max_value = rule.get("max_value")
    if max_value is not None and value > float(max_value):
        failures.append(f"{value:.6g} > {float(max_value):.6g}")
    min_abs_value = rule.get("min_abs_value")
    if min_abs_value is not None and abs(value) < float(min_abs_value):
        failures.append(f"abs({value:.6g}) < {float(min_abs_value):.6g}")
    return RunMetricResult(
        metric=metric,
        value=value,
        passed=not failures,
        reason="; ".join(failures) if failures else "within threshold",
    )


def check_run(
    run_path: Path,
    policy: Mapping[str, Any],
) -> tuple[list[RunMetricResult], int, bool]:
    steps, complete = load_run(run_path)
    warmup = int(policy.get("warmup_steps", 0))
    minimum = int(policy.get("minimum_steps", 1))
    steady_steps = steps[warmup:]
    expected = policy.get("expected_steps")
    if expected is not None and len(steady_steps) != int(expected):
        raise ValueError(
            f"unexpected steady-state step count: found={len(steady_steps)}, "
            f"expected={int(expected)}"
        )
    if len(steady_steps) < minimum:
        raise ValueError(f"not enough steady-state steps: found={len(steady_steps)}, required={minimum}")
    if bool(policy.get("require_complete_run", True)) and not complete:
        raise ValueError("experiment record is incomplete: expected run_start ... run_end")

    results = []
    for metric, raw_rule in policy.get("metrics", {}).items():
        rule = dict(raw_rule or {})
        reducer = str(rule.get("reducer", "min"))
        results.append(
            check_metric(
                str(metric),
                reduce_metric(steady_steps, str(metric), reducer),
                rule,
            )
        )
    return results, len(steady_steps), complete


def render_markdown(results: list[RunMetricResult]) -> str:
    lines = [
        "| metric | value | gate |",
        "|---|---:|---|",
    ]
    for result in results:
        value = "missing" if result.value is None else f"{result.value:.6g}"
        gate = "PASS" if result.passed else f"FAIL: {result.reason}"
        lines.append(f"| {result.metric} | {value} | {gate} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    results, steady_steps, complete = check_run(args.run, policy)
    passed = all(result.passed for result in results)
    print(render_markdown(results))
    print(f"\nsteady-state steps: {steady_steps}; complete: {complete}")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(
                {
                    "passed": passed,
                    "steady_steps": steady_steps,
                    "complete": complete,
                    "metrics": [asdict(result) for result in results],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
