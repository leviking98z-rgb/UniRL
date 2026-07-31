import json

import pytest

from benchmarks.framework.compare_runs import compare_runs, load_steps


def _write_run(path, rows):
    with path.open("w") as handle:
        for step, metrics in enumerate(rows, 1):
            handle.write(json.dumps({"record_type": "step", "step": step, "metrics": metrics}) + "\n")


def test_compare_runs_passes_faster_candidate(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_run(
        baseline,
        [{"perf/step_time_s": 10.0, "rollout/reward_mean": 0.5} for _ in range(5)],
    )
    _write_run(
        candidate,
        [{"perf/step_time_s": 9.0, "rollout/reward_mean": 0.49} for _ in range(5)],
    )
    policy = {
        "warmup_steps": 1,
        "minimum_steps": 4,
        "metrics": {
            "perf/step_time_s": {"direction": "lower", "max_regression_pct": 3},
            "rollout/reward_mean": {"direction": "higher", "max_regression_abs": 0.02},
        },
    }

    results, baseline_steps, candidate_steps = compare_runs(baseline, candidate, policy)
    assert baseline_steps == candidate_steps == 4
    assert all(result.passed for result in results)


def test_compare_runs_fails_reward_regression_and_missing_required_metric(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_run(baseline, [{"rollout/reward_mean": 0.5, "perf/max_memory_allocated_gb": 10.0}] * 3)
    _write_run(candidate, [{"rollout/reward_mean": 0.4}] * 3)
    policy = {
        "minimum_steps": 3,
        "warmup_steps": 0,
        "metrics": {
            "rollout/reward_mean": {"direction": "higher", "max_regression_abs": 0.02},
            "perf/max_memory_allocated_gb": {"direction": "lower", "required": True},
        },
    }

    results, _, _ = compare_runs(baseline, candidate, policy)
    assert [result.passed for result in results] == [False, False]


def test_load_steps_rejects_invalid_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{bad json}\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_steps(path)
