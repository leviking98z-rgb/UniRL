"""Compare two UniRL structured experiment records and enforce regression gates."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


@dataclass(frozen=True)
class MetricResult:
    metric: str
    baseline: Optional[float]
    candidate: Optional[float]
    regression_pct: Optional[float]
    regression_abs: Optional[float]
    passed: bool
    reason: str


def load_steps(path: Path) -> list[Dict[str, float]]:
    steps: list[Dict[str, float]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if record.get("record_type") != "step":
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
    return steps


def reduce_metric(steps: Iterable[Mapping[str, float]], metric: str, reducer: str) -> Optional[float]:
    values = [float(step[metric]) for step in steps if metric in step]
    if not values:
        return None
    if reducer == "median":
        return statistics.median(values)
    if reducer == "mean":
        return statistics.mean(values)
    if reducer == "max":
        return max(values)
    if reducer == "min":
        return min(values)
    raise ValueError(f"unsupported reducer {reducer!r} for {metric}")


def compare_metric(
    metric: str,
    baseline: Optional[float],
    candidate: Optional[float],
    rule: Mapping[str, Any],
) -> MetricResult:
    required = bool(rule.get("required", True))
    if baseline is None or candidate is None:
        passed = not required
        return MetricResult(metric, baseline, candidate, None, None, passed, "missing metric")

    direction = str(rule.get("direction", "lower"))
    if direction == "lower":
        regression_abs = candidate - baseline
    elif direction == "higher":
        regression_abs = baseline - candidate
    else:
        raise ValueError(f"{metric}: direction must be 'lower' or 'higher'")

    regression_pct = None
    if baseline != 0:
        regression_pct = 100.0 * regression_abs / abs(baseline)

    failures = []
    max_pct = rule.get("max_regression_pct")
    if max_pct is not None and regression_pct is not None and regression_pct > float(max_pct):
        failures.append(f"{regression_pct:.2f}% > {float(max_pct):.2f}%")
    max_abs = rule.get("max_regression_abs")
    if max_abs is not None and regression_abs > float(max_abs):
        failures.append(f"{regression_abs:.6g} > {float(max_abs):.6g} absolute")
    return MetricResult(
        metric,
        baseline,
        candidate,
        regression_pct,
        regression_abs,
        not failures,
        "; ".join(failures) if failures else "within threshold",
    )


def compare_runs(
    baseline_path: Path,
    candidate_path: Path,
    policy: Mapping[str, Any],
) -> tuple[list[MetricResult], int, int]:
    warmup = int(policy.get("warmup_steps", 2))
    minimum = int(policy.get("minimum_steps", 5))
    baseline_steps = load_steps(baseline_path)[warmup:]
    candidate_steps = load_steps(candidate_path)[warmup:]
    if len(baseline_steps) < minimum or len(candidate_steps) < minimum:
        raise ValueError(
            f"not enough steady-state steps: baseline={len(baseline_steps)}, "
            f"candidate={len(candidate_steps)}, required={minimum}"
        )

    results = []
    for metric, raw_rule in policy.get("metrics", {}).items():
        rule = dict(raw_rule or {})
        reducer = str(rule.get("reducer", "median"))
        results.append(
            compare_metric(
                str(metric),
                reduce_metric(baseline_steps, str(metric), reducer),
                reduce_metric(candidate_steps, str(metric), reducer),
                rule,
            )
        )
    return results, len(baseline_steps), len(candidate_steps)


def render_markdown(results: Iterable[MetricResult]) -> str:
    lines = [
        "| metric | baseline | candidate | regression | gate |",
        "|---|---:|---:|---:|---|",
    ]
    for result in results:
        baseline = "missing" if result.baseline is None else f"{result.baseline:.6g}"
        candidate = "missing" if result.candidate is None else f"{result.candidate:.6g}"
        regression = "n/a" if result.regression_pct is None else f"{result.regression_pct:+.2f}%"
        gate = "PASS" if result.passed else f"FAIL: {result.reason}"
        lines.append(f"| {result.metric} | {baseline} | {candidate} | {regression} | {gate} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(__file__).with_name("thresholds.yaml"),
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    results, baseline_steps, candidate_steps = compare_runs(args.baseline, args.candidate, policy)
    print(render_markdown(results))
    print(f"\nsteady-state steps: baseline={baseline_steps}, candidate={candidate_steps}")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(
                {
                    "passed": all(result.passed for result in results),
                    "baseline_steps": baseline_steps,
                    "candidate_steps": candidate_steps,
                    "metrics": [result.__dict__ for result in results],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    raise SystemExit(0 if all(result.passed for result in results) else 1)


if __name__ == "__main__":
    main()
