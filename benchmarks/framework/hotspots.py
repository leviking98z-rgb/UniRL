"""Rank where a run's wall-clock actually goes, and what could still be won.

The candidate-generation half of the loop. ``check_run`` says a run was healthy
and ``effect_size`` says whether a change helped; neither says *what to try
next*. Guessing that wastes cluster time: the previous iteration spent runs on
reward-path and startup work that could not have mattered, because reward was
0.9% of the step and startup does not touch steady state at all.

Two things this prints that a raw metric dump does not:

* **Share of step, sorted.** An optimization's ceiling is the phase's share. A
  2x win on a 1% phase is 0.5%, which is inside this cluster's noise floor and
  therefore unmeasurable, let alone worth a config change.
* **Idle-GPU accounting.** When phases run on disjoint device sets (HI3 pins the
  AR engine to GPUs 0-3 and the DiT engine to 4-7), a serial phase order leaves
  half the cluster idle. That idle time is invisible in a per-phase table but is
  usually the largest single number available.

Amdahl bound per phase is reported as the step-level gain from removing the
phase ENTIRELY, which no real change achieves — it is the ceiling that rules
candidates out, not a target.

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from benchmarks.framework.compare_runs import load_steps

# Phases whose device sets are disjoint in the HI3 two-engine recipe: the AR
# engine owns GPUs 0-3 and the DiT engine owns 4-7 (a real partition, pinned by
# each stage YAML's runtime.devices). While one generates the other's cards are
# idle, so a serial order caps utilization at ~50% for the whole rollout.
DISJOINT_DEVICE_PHASES = ("perf/ar_generate_time_s", "perf/image_generate_time_s")

# Total-time metric that the shares are taken against.
STEP_METRIC = "perf/step_time_s"

# Metrics that are gauges (memory, counts), not durations.
NON_DURATION_SUFFIXES = ("_gb", "_count", "_ratio", "_norm")


@dataclass(frozen=True)
class Phase:
    metric: str
    seconds: float
    share_pct: float
    amdahl_ceiling_pct: float

    @property
    def measurable(self) -> bool:
        """Whether removing this phase entirely would clear a 5% effect gate.

        Phases below this cannot produce an acceptable result no matter how good
        the optimization, so they are not worth a cluster run.
        """
        return self.amdahl_ceiling_pct >= 5.0


def is_duration(metric: str) -> bool:
    return metric.startswith("perf/") and metric.endswith("_s") and not metric.endswith(NON_DURATION_SUFFIXES)


def collect_phases(steps: Sequence[Mapping[str, float]]) -> tuple[List[Phase], float]:
    """Median each duration metric across steps and rank by share of step time."""
    if not steps:
        raise ValueError("no step records found")

    def median_of(metric: str) -> Optional[float]:
        values = [float(s[metric]) for s in steps if metric in s]
        return st.median(values) if values else None

    step_time = median_of(STEP_METRIC)
    if not step_time:
        raise ValueError(f"{STEP_METRIC} missing or zero; cannot compute shares")

    metrics = {key for step in steps for key in step if is_duration(key)}
    phases: List[Phase] = []
    for metric in sorted(metrics):
        if metric == STEP_METRIC:
            continue
        seconds = median_of(metric)
        if seconds is None:
            continue
        share = 100.0 * seconds / step_time
        phases.append(
            Phase(
                metric=metric,
                seconds=seconds,
                share_pct=share,
                # Removing the phase outright: the step shrinks by its share.
                amdahl_ceiling_pct=share,
            )
        )
    phases.sort(key=lambda p: -p.seconds)
    return phases, step_time


def idle_device_findings(phases: Sequence[Phase], step_time: float) -> List[Dict[str, Any]]:
    """Report serial phases that hold disjoint device sets.

    Two phases pinned to different GPUs, run one after the other, means each
    phase's cards sit idle for the other's duration. Overlapping them is bounded
    below by the longer phase, so the recoverable time is the shorter one.
    """
    by_metric = {p.metric: p for p in phases}
    present = [by_metric[m] for m in DISJOINT_DEVICE_PHASES if m in by_metric]
    if len(present) < 2:
        return []
    serial = sum(p.seconds for p in present)
    floor = max(p.seconds for p in present)
    recoverable = serial - floor
    return [
        {
            "kind": "serial_disjoint_devices",
            "phases": [p.metric for p in present],
            "serial_seconds": serial,
            "overlap_floor_seconds": floor,
            "recoverable_seconds": recoverable,
            "recoverable_step_pct": 100.0 * recoverable / step_time,
            "note": (
                "phases hold disjoint GPU sets but run serially; each set idles for the other's "
                "duration. Perfect overlap is bounded below by the longer phase, so this is the "
                "ceiling, not a target — sub-batch pipelining reaches a fraction of it."
            ),
        }
    ]


def render(phases: Sequence[Phase], step_time: float, findings: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        f"step_time_s (median): {step_time:.2f}",
        "",
        "| phase | seconds | share of step | ceiling if removed | worth a run? |",
        "|---|---:|---:|---:|---|",
    ]
    for p in phases:
        if p.seconds < 0.01:
            continue
        worth = "yes" if p.measurable else "no (below 5% gate)"
        lines.append(f"| {p.metric} | {p.seconds:.2f} | {p.share_pct:.1f}% | {p.amdahl_ceiling_pct:.1f}% | {worth} |")

    accounted = sum(p.seconds for p in phases if _is_top_level(p.metric))
    coverage = 100.0 * accounted / step_time
    lines += [
        "",
        f"top-level phases sum to {accounted:.2f}s of {step_time:.2f}s ({coverage:.1f}%).",
    ]
    if coverage > 100.5:
        # Phase timers are entered/exited independently of the step timer, so a
        # small overshoot is instrumentation overlap, not double counting. Say so
        # rather than letting a >100% figure read as a decomposition error.
        lines.append(
            f"  note: {coverage - 100.0:.1f}% over — phase timers overlap the step timer's "
            "boundaries; treat shares as approximate, not a partition."
        )
    elif coverage < 95.0:
        lines.append(f"  note: {100.0 - coverage:.1f}% of the step is unattributed by any phase timer.")
    if findings:
        lines += ["", "structural findings:"]
        for f in findings:
            lines.append(
                f"  - {f['kind']}: {' + '.join(f['phases'])} run serially "
                f"({f['serial_seconds']:.1f}s) but could not fall below {f['overlap_floor_seconds']:.1f}s; "
                f"up to {f['recoverable_seconds']:.1f}s = {f['recoverable_step_pct']:.1f}% of step is recoverable"
            )
            lines.append(f"    {f['note']}")
    return "\n".join(lines)


def _is_top_level(metric: str) -> bool:
    """``perf/train/x_s`` is a breakdown of ``perf/train_time_s``; don't double-count."""
    return metric.count("/") == 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="experiment.jsonl from a healthy steady-state run")
    parser.add_argument("--warmup", type=int, default=0, help="steps to drop before taking medians")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    steps = load_steps(args.run)[args.warmup :]
    phases, step_time = collect_phases(steps)
    findings = idle_device_findings(phases, step_time)
    print(render(phases, step_time, findings))

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(
                {
                    "step_time_s": step_time,
                    "steps_used": len(steps),
                    "phases": [
                        {
                            "metric": p.metric,
                            "seconds": p.seconds,
                            "share_pct": p.share_pct,
                            "amdahl_ceiling_pct": p.amdahl_ceiling_pct,
                            "worth_a_run": p.measurable,
                        }
                        for p in phases
                    ],
                    "findings": list(findings),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
