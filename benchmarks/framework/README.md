# Framework performance loop

This directory turns UniRL's common trainer telemetry into a repeatable
candidate gate:

1. Run a baseline and candidate with the same workload. Set
   `UNIRL_EXPERIMENT_OUTPUT=/path/to/experiment.jsonl` (or
   `logging.experiment_output=...`) to retain step-level metrics without WandB.
   Existing files are rejected to prevent accidental run mixing; set
   `logging.experiment_append=true` only for an intentional checkpoint resume.
2. Drop warmup and compare the structured records:

   ```bash
   python -m benchmarks.framework.compare_runs \
     baseline.jsonl candidate.jsonl \
   --policy benchmarks/framework/thresholds.yaml
   ```

3. A non-zero exit means at least one speed, memory, or reward guard regressed.

`run_matrix.py` expands representative diffusion, AR, and unified-model
workloads. It is dry-run by default:

```bash
python -m benchmarks.framework.run_matrix --tag diffusion
python -m benchmarks.framework.run_matrix --workload sd3_flowgrpo_vllmomni_1x8 \
  --output-dir framework_results/candidate --execute
```

The matrix runner intentionally does not claim cluster nodes or change source
revisions. Cluster orchestration remains outside the training process; run it
only after the node lifecycle has been handled.

The checked-in `.yaml` policy and matrix use JSON syntax (a YAML subset), so
these control-plane tools need only the Python standard library.
