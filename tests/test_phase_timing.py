import importlib.util
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).parents[1] / "unirl" / "utils" / "timing.py"
SPEC = importlib.util.spec_from_file_location("unirl_timing_under_test", MODULE_PATH)
timing = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(timing)

PhaseTimings = timing.PhaseTimings
critical_path_phase_times = timing.critical_path_phase_times


def test_phase_timings_accumulates_reentered_phase():
    with patch.object(timing.time, "perf_counter", side_effect=[1.0, 3.0, 4.0, 7.5]):
        timings = PhaseTimings()
        with timings.measure("train"):
            pass
        with timings.measure("train"):
            pass

    assert timings.get("train") == 5.5


def test_critical_path_phase_times_uses_slowest_worker_and_prefix():
    phases = critical_path_phase_times(
        {
            "ar_backward": [3.0, 4.5, 4.0, 3.5],
            "image_backward": [7.0, 6.0, 8.0, 7.5],
            "empty": [],
        },
        prefix="train/",
    )

    assert phases == {
        "train/ar_backward": 4.5,
        "train/image_backward": 8.0,
    }
