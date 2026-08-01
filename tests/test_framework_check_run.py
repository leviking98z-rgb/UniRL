import json

import pytest

from benchmarks.framework.check_run import check_run
from benchmarks.framework.run_matrix import build_health_command


def _write_run(path, metrics, *, complete=True):
    records = [
        {"record_type": "run_start"},
        {"record_type": "step", "step": 1, "metrics": metrics},
    ]
    if complete:
        records.append({"record_type": "run_end"})
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_check_run_accepts_nonzero_learning_signal(tmp_path):
    run = tmp_path / "run.jsonl"
    _write_run(
        run,
        {
            "rollout/ar_advantage_std": 0.8,
            "train/ar/has_backward": 1.0,
            "train/ar/grad_norm": 0.04,
            "train/ar/loss": -0.01,
        },
    )
    policy = {
        "metrics": {
            "rollout/ar_advantage_std": {"min_value": 1e-6},
            "train/ar/has_backward": {"min_value": 1.0},
            "train/ar/grad_norm": {"min_value": 1e-8},
            "train/ar/loss": {"min_abs_value": 1e-8},
        }
    }

    results, steps, complete = check_run(run, policy)
    assert steps == 1
    assert complete
    assert all(result.passed for result in results)


def test_check_run_rejects_degenerate_learning_signal(tmp_path):
    run = tmp_path / "run.jsonl"
    _write_run(
        run,
        {
            "rollout/ar_advantage_std": 0.0,
            "train/ar/has_backward": 1.0,
            "train/ar/grad_norm": 0.0,
        },
    )
    policy = {
        "metrics": {
            "rollout/ar_advantage_std": {"min_value": 1e-6},
            "train/ar/has_backward": {"min_value": 1.0},
            "train/ar/grad_norm": {"min_value": 1e-8},
        }
    }

    results, _, _ = check_run(run, policy)
    assert [result.passed for result in results] == [False, True, False]


def test_check_run_rejects_missing_run_end(tmp_path):
    run = tmp_path / "run.jsonl"
    _write_run(run, {"train/ar/has_backward": 1.0}, complete=False)

    with pytest.raises(ValueError, match="incomplete"):
        check_run(run, {"metrics": {}})


def test_check_run_rejects_partial_run_with_run_end(tmp_path):
    run = tmp_path / "run.jsonl"
    _write_run(run, {"train/ar/has_backward": 1.0})

    with pytest.raises(ValueError, match="step count.*found=1, expected=3"):
        check_run(run, {"expected_steps": 3, "metrics": {}})


def test_matrix_health_command_is_opt_in(tmp_path):
    experiment = tmp_path / "experiment.jsonl"
    output = tmp_path / "health.json"

    assert build_health_command({}, experiment, output) is None
    command = build_health_command(
        {"health_policy": "benchmarks/framework/hi3_learning_thresholds.yaml"},
        experiment,
        output,
    )
    assert command is not None
    assert command[-5:] == [
        str(experiment),
        "--policy",
        "benchmarks/framework/hi3_learning_thresholds.yaml",
        "--json-output",
        str(output),
    ]
