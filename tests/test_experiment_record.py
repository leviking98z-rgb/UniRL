import json

import pytest

from unirl.utils.experiment_record import ExperimentRecorder


def test_experiment_recorder_writes_run_and_step(tmp_path):
    output = tmp_path / "nested" / "experiment.jsonl"
    recorder = ExperimentRecorder(str(output), run_name="candidate", metadata={"gpus": 8})
    recorder.log_step(3, {"perf/step_time_s": 4.5})
    recorder.finish()

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["record_type"] for record in records] == ["run_start", "step", "run_end"]
    assert records[0]["metadata"] == {"gpus": 8}
    assert records[1]["step"] == 3
    assert records[1]["metrics"]["perf/step_time_s"] == 4.5


def test_experiment_recorder_is_disabled_without_output():
    recorder = ExperimentRecorder(None)
    recorder.log_step(1, {"ignored": 1})
    recorder.finish()
    assert not recorder.enabled


def test_experiment_recorder_rejects_existing_output_without_append(tmp_path):
    output = tmp_path / "experiment.jsonl"
    output.write_text("")
    with pytest.raises(FileExistsError):
        ExperimentRecorder(str(output))
