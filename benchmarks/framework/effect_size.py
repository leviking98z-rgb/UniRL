"""Decide whether a speed delta is real or noise, given replicate runs.

``compare_runs`` answers "did metric X move more than N%?" against a single
baseline and a single candidate. That question is unanswerable at N below the
noise floor, and on this cluster the floor is not small: three identical
8-GPU HI3 configs produced train times of 137.4s, 137.9s and 149.2s — an 8.6%
spread. A 3% single-pair gate labels 4 of those 6 orderings as a real
improvement or regression. Every one of them is noise.

This module takes REPLICATES per arm and reports an effect only when it clears
the observed within-arm variation:

  * Welch's t (unequal variances, no normality assumption beyond the CLT) with a
    two-sided p-value from a Student-t survival function computed in the stdlib.
  * A bootstrap percentile CI on the relative delta, so the report carries an
    interval rather than a point estimate.
  * A minimum-replicates guard: n=1 per arm cannot separate effect from noise,
    so it is reported as ``undetermined`` instead of a number.

A candidate is ACCEPTED only when the CI's near side still clears
``min_effect_pct``. That is deliberately stricter than "p < 0.05": a
statistically detectable 1% win is not worth a config change, and a 20% point
estimate whose CI spans zero is not a win at all.

Stdlib only (no scipy/numpy) — the control plane must run on any cluster node.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from benchmarks.framework.compare_runs import load_steps, reduce_metric

# Deterministic bootstrap: the same inputs must yield the same verdict, or a
# gate becomes unreproducible and re-running it becomes a way to change it.
BOOTSTRAP_SEED = 20260803
BOOTSTRAP_RESAMPLES = 10000


@dataclass(frozen=True)
class ArmSummary:
    """One config's per-run reduced values."""

    name: str
    values: List[float]

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> Optional[float]:
        """None when the metric is absent from every run — a workload that never
        emits it must render as "missing", not raise."""
        return st.mean(self.values) if self.values else None

    @property
    def sd(self) -> Optional[float]:
        return st.stdev(self.values) if self.n > 1 else None

    @property
    def cv_pct(self) -> Optional[float]:
        sd = self.sd
        mean = self.mean
        if sd is None or not mean:
            return None
        return 100.0 * sd / abs(mean)


@dataclass(frozen=True)
class EffectResult:
    metric: str
    baseline: ArmSummary
    candidate: ArmSummary
    delta_pct: Optional[float]
    ci_low_pct: Optional[float]
    ci_high_pct: Optional[float]
    p_value: Optional[float]
    verdict: str
    reason: str


def _student_t_sf(t: float, df: float) -> float:
    """Upper-tail P(T > t) for Student's t, via the regularized incomplete beta.

    ``statistics.NormalDist`` would ignore the small-sample tail that matters
    most here (n=3 per arm is df~4), so the exact t distribution is used.
    """
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    # P(|T| > t) = I_x(df/2, 1/2); halve for the one-sided upper tail.
    return 0.5 * _betainc_regularized(df / 2.0, 0.5, x)


def _betainc_regularized(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) by continued fraction (Lentz)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # Use the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) on the slow-converging side.
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc_regularized(b, a, 1.0 - x)
    log_prefix = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    tiny = 1e-30
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        # even step
        aa = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        # odd step
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return math.exp(log_prefix) * h / a


def welch_p_value(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Two-sided Welch's t-test p-value; None when either arm has n<2 or zero spread."""
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = st.variance(a), st.variance(b)
    na, nb = len(a), len(b)
    se2 = va / na + vb / nb
    if se2 <= 0.0:
        return None
    t = (st.mean(b) - st.mean(a)) / math.sqrt(se2)
    # Welch-Satterthwaite degrees of freedom.
    denom = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    if denom <= 0.0:
        return None
    df = se2 * se2 / denom
    return min(1.0, 2.0 * _student_t_sf(abs(t), df))


def bootstrap_delta_ci(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[Optional[float], Optional[float]]:
    """Percentile CI for 100*(mean(candidate)-mean(baseline))/mean(baseline)."""
    if len(baseline) < 2 or len(candidate) < 2:
        return None, None
    rng = random.Random(seed)
    deltas: List[float] = []
    for _ in range(resamples):
        rb = st.mean(rng.choices(baseline, k=len(baseline)))
        rc = st.mean(rng.choices(candidate, k=len(candidate)))
        if rb == 0:
            continue
        deltas.append(100.0 * (rc - rb) / abs(rb))
    if not deltas:
        return None, None
    deltas.sort()
    tail = (1.0 - confidence) / 2.0
    low = deltas[max(0, int(math.floor(tail * len(deltas))) - 1)]
    high = deltas[min(len(deltas) - 1, int(math.ceil((1.0 - tail) * len(deltas))))]
    return low, high


def decide(
    metric: str,
    baseline: ArmSummary,
    candidate: ArmSummary,
    rule: Mapping[str, Any],
) -> EffectResult:
    """Classify one metric as improvement / regression / neutral / undetermined.

    ``direction='lower'`` (default) means smaller is better, as for step time.
    """
    direction = str(rule.get("direction", "lower"))
    if direction not in ("lower", "higher"):
        raise ValueError(f"{metric}: direction must be 'lower' or 'higher'")
    min_effect = float(rule.get("min_effect_pct", 5.0))
    alpha = float(rule.get("alpha", 0.05))
    min_replicates = int(rule.get("min_replicates", 3))

    if not baseline.values or not candidate.values:
        return EffectResult(metric, baseline, candidate, None, None, None, None, "undetermined", "metric missing")

    delta = 100.0 * (candidate.mean - baseline.mean) / abs(baseline.mean) if baseline.mean else None

    if baseline.n < min_replicates or candidate.n < min_replicates:
        return EffectResult(
            metric,
            baseline,
            candidate,
            delta,
            None,
            None,
            None,
            "undetermined",
            f"need >={min_replicates} replicates per arm; got baseline={baseline.n}, candidate={candidate.n}",
        )

    p = welch_p_value(baseline.values, candidate.values)
    ci_low, ci_high = bootstrap_delta_ci(baseline.values, candidate.values)

    # Orient so that "gain" is always positive-good.
    gain = -delta if direction == "lower" else delta
    gain_ci_near = None
    if ci_low is not None and ci_high is not None:
        gain_ci_near = -ci_high if direction == "lower" else ci_low

    if p is not None and p > alpha:
        return EffectResult(
            metric, baseline, candidate, delta, ci_low, ci_high, p, "neutral", f"p={p:.3g} > alpha={alpha}"
        )
    if gain_ci_near is None:
        return EffectResult(
            metric, baseline, candidate, delta, ci_low, ci_high, p, "undetermined", "no CI (degenerate spread)"
        )
    if gain_ci_near >= min_effect:
        return EffectResult(
            metric,
            baseline,
            candidate,
            delta,
            ci_low,
            ci_high,
            p,
            "improvement",
            f"CI near side {gain_ci_near:+.2f}% clears min_effect={min_effect}%",
        )
    if -gain_ci_near >= min_effect and gain < 0:
        return EffectResult(
            metric,
            baseline,
            candidate,
            delta,
            ci_low,
            ci_high,
            p,
            "regression",
            f"CI shows a loss of at least {-gain_ci_near:.2f}%",
        )
    return EffectResult(
        metric,
        baseline,
        candidate,
        delta,
        ci_low,
        ci_high,
        p,
        "neutral",
        f"CI near side {gain_ci_near:+.2f}% below min_effect={min_effect}%",
    )


def summarize_arm(name: str, paths: Sequence[Path], metric: str, reducer: str, warmup: int) -> ArmSummary:
    """One reduced value per run — runs are the replicate unit, not steps.

    Steps inside a run share caches, allocator state and a compile, so they are
    not independent samples of a config's speed. Averaging steps inside a run
    and treating RUNS as replicates is what makes the variance estimate honest.
    """
    values: List[float] = []
    for path in paths:
        steps = load_steps(path)[warmup:]
        if not steps:
            continue
        value = reduce_metric(steps, metric, reducer)
        if value is not None:
            values.append(float(value))
    return ArmSummary(name=name, values=values)


def load_provenance(path: Path) -> Dict[str, Any]:
    """The ``run_start`` git / environment block, or {} for a pre-schema-2 record."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") == "run_start":
                return {
                    "git": record.get("git") or {},
                    "environment": record.get("environment") or {},
                }
    return {}


def check_comparability(baseline: Sequence[Path], candidate: Sequence[Path]) -> List[str]:
    """Warn when the two arms differ in anything other than the knob under test.

    Returns human-readable warnings; an empty list means the arms look
    comparable. This exists because attributing a delta to a code change
    requires that nothing ELSE moved: the prior iteration of this framework
    compared runs built from different revisions and reported the difference as
    an optimization result.
    """
    warnings: List[str] = []
    all_paths = list(baseline) + list(candidate)
    provenance = {path: load_provenance(path) for path in all_paths}

    missing = [p.name for p, v in provenance.items() if not v.get("git", {}).get("commit")]
    if missing:
        warnings.append(f"no commit recorded (pre-schema-2 records): {missing}")

    commits = {v["git"].get("commit") for v in provenance.values() if v.get("git", {}).get("commit")}
    if len(commits) > 1:
        warnings.append(f"arms span {len(commits)} commits: {sorted(c[:8] for c in commits if c)}")

    dirty = [p.name for p, v in provenance.items() if v.get("git", {}).get("dirty")]
    if dirty:
        warnings.append(f"uncommitted changes during run (delta not attributable): {dirty}")

    # Fail closed: a dirty check that could not complete (slow network mount) is
    # NOT evidence of a clean tree. Only an explicit False clears this.
    unverified = [
        p.name
        for p, v in provenance.items()
        if v.get("git", {}).get("commit") and v.get("git", {}).get("dirty") is None
    ]
    if unverified:
        warnings.append(f"could not verify tree was clean (dirty=null): {unverified}")

    for key in ("device_name", "device_count", "torch", "cuda"):
        seen = {v["environment"].get(key) for v in provenance.values() if v.get("environment", {}).get(key)}
        if len(seen) > 1:
            warnings.append(f"arms differ in {key}: {sorted(str(s) for s in seen)}")

    return warnings


def render_markdown(results: Sequence[EffectResult]) -> str:
    lines = [
        "| metric | baseline (n, cv) | candidate (n, cv) | delta | 95% CI | p | verdict |",
        "|---|---|---|---:|---|---:|---|",
    ]
    for r in results:

        def arm(a: ArmSummary) -> str:
            if a.mean is None:
                return "missing"
            cv = "n/a" if a.cv_pct is None else f"{a.cv_pct:.1f}%"
            return f"{a.mean:.6g} (n={a.n}, cv={cv})"

        delta = "n/a" if r.delta_pct is None else f"{r.delta_pct:+.2f}%"
        ci = "n/a" if r.ci_low_pct is None else f"[{r.ci_low_pct:+.2f}%, {r.ci_high_pct:+.2f}%]"
        p = "n/a" if r.p_value is None else f"{r.p_value:.3g}"
        lines.append(f"| {r.metric} | {arm(r.baseline)} | {arm(r.candidate)} | {delta} | {ci} | {p} | {r.verdict} |")
    return "\n".join(lines)


def _arm_summary_dict(a: ArmSummary) -> Dict[str, Any]:
    return {"name": a.name, "n": a.n, "mean": a.mean, "sd": a.sd, "cv_pct": a.cv_pct, "values": a.values}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, action="append", required=True, help="repeat per replicate run")
    parser.add_argument("--candidate", type=Path, action="append", required=True, help="repeat per replicate run")
    parser.add_argument("--policy", type=Path, default=Path(__file__).with_name("effect_thresholds.yaml"))
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--allow-mismatched-arms",
        action="store_true",
        help="downgrade comparability violations (differing commit/host/dirty tree) to warnings",
    )
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    warmup = int(policy.get("warmup_steps", 0))

    comparability = check_comparability(args.baseline, args.candidate)
    if comparability:
        print("comparability warnings:")
        for warning in comparability:
            print(f"  - {warning}")
        print()

    results: List[EffectResult] = []
    for metric, raw_rule in policy.get("metrics", {}).items():
        rule = dict(raw_rule or {})
        reducer = str(rule.get("reducer", "median"))
        results.append(
            decide(
                str(metric),
                summarize_arm(args.baseline_name, args.baseline, str(metric), reducer, warmup),
                summarize_arm(args.candidate_name, args.candidate, str(metric), reducer, warmup),
                rule,
            )
        )

    print(render_markdown(results))

    # Two distinct roles, previously conflated under one "gating" flag:
    #   role=objective — at least one must improve, or there is nothing to accept.
    #   role=guard     — must not regress, but is EXPECTED to stay flat. Reward is
    #                    the canonical guard: a speed change that moves reward is
    #                    suspect, and requiring reward to IMPROVE would reject
    #                    every honest speed optimization.
    # Default role is "objective" when gating is on, so existing policies behave
    # as before.
    def role_of(metric: str) -> str:
        rule = policy.get("metrics", {}).get(metric, {})
        if not bool(rule.get("gating", True)):
            return "report"
        return str(rule.get("role", "objective"))

    objectives = [r for r in results if role_of(r.metric) == "objective"]
    guards = [r for r in results if role_of(r.metric) == "guard"]

    regressions = [r for r in objectives + guards if r.verdict == "regression"]
    # An undetermined GUARD is a real blocker (we cannot show it stayed flat);
    # an undetermined objective means the win itself is unproven.
    undetermined = [r for r in objectives + guards if r.verdict == "undetermined"]
    improvements = [r for r in objectives if r.verdict == "improvement"]
    # Comparability blocks acceptance by default: an effect measured across two
    # commits or two hosts is not attributable to the knob under test, however
    # clean its statistics look.
    blocked = bool(comparability) and not args.allow_mismatched_arms
    accepted = bool(improvements) and not regressions and not undetermined and not blocked

    print()
    print(f"improvements={[r.metric for r in improvements]}")
    print(f"regressions={[r.metric for r in regressions]}")
    print(f"undetermined={[r.metric for r in undetermined]}")
    print(f"guards_held={[r.metric for r in guards if r.verdict in ('neutral', 'improvement')]}")
    if blocked:
        print("BLOCKED: arms are not comparable (pass --allow-mismatched-arms to override)")
    print(f"ACCEPT={accepted}")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(
                {
                    "accepted": accepted,
                    "comparability_warnings": comparability,
                    "comparability_blocked": blocked,
                    "metrics": [
                        {
                            **{k: v for k, v in asdict(r).items() if k not in ("baseline", "candidate")},
                            "baseline": _arm_summary_dict(r.baseline),
                            "candidate": _arm_summary_dict(r.candidate),
                        }
                        for r in results
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    raise SystemExit(0 if accepted else 1)


if __name__ == "__main__":
    main()
