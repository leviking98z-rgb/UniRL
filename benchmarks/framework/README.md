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

For a one-run learning smoke test, use the absolute health gate. It rejects
records that merely finish while producing zero advantages or zero gradients:

```bash
python -m benchmarks.framework.check_run \
  experiment.jsonl \
  --policy benchmarks/framework/hi3_learning_thresholds.yaml
```

The HI3 learning workload intentionally samples two AR recaptions and two
images per recaption. This yields four joint trajectories per prompt, so GRPO
has within-group variation on both tracks. When a workload declares
`health_policy`, `run_matrix.py --execute` runs that gate automatically after
training and fails the workload if the record is incomplete or degenerate.

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

## Deciding whether a speedup is real

`compare_runs.py` answers "did metric X move more than N%?" between ONE baseline
and ONE candidate. That question is unanswerable below the noise floor, and on
this cluster the floor is not small. Three identical 8-GPU HI3 configs measured
137.4s, 137.9s and 149.2s of train time — an 8.6% spread. A 3% single-pair gate
labels **4 of those 6 orderings** as a real improvement or regression. All six
are the same config.

`effect_size.py` requires replicates and reports an effect only when it clears
the observed within-arm variation:

```bash
# 3 replicates per arm, differing ONLY in the knob under test
python -m benchmarks.framework.run_matrix --workload hi3_unified_learning_smoke_1x8 \
  --replicates 3 --output-dir results/base --execute
python -m benchmarks.framework.run_matrix --workload hi3_unified_learning_smoke_1x8 \
  --replicates 3 --output-dir results/cand --execute \
  --override bundle.config.batch_replay_steps=true

python -m benchmarks.framework.effect_size \
  --baseline results/base/hi3_unified_learning_smoke_1x8/replicate-{0,1,2}/experiment.jsonl \
  --candidate results/cand/hi3_unified_learning_smoke_1x8/replicate-{0,1,2}/experiment.jsonl
```

(the shell brace form expands to three `--baseline` flags; repeat the flag
explicitly if your shell does not.)

It reports Welch's t p-value plus a deterministic bootstrap CI on the relative
delta, and accepts only when the CI's near side still clears
`min_effect_pct` (default 5%). Consequences worth knowing:

- **n<3 per arm reports `undetermined`, not a number.** One run cannot separate
  effect from noise.
- **A statistically significant 1% win is `neutral`.** Detectable is not the
  same as worth a config change.
- **A 20% point estimate whose CI spans zero is not a win.** This is what
  rejects the kind of single-run number that reads as a result and is not one.

Replicates are separate directories, never separate steps in one file: steps
inside a run share caches, allocator state and a compile, so they are not
independent samples of a config's speed. Runs are the replicate unit.

### Comparability is enforced, not assumed

A delta is only attributable to the knob under test if nothing else moved. Each
`run_start` record carries `git` (commit, dirty, branch) and `environment`
(host, torch, CUDA, device name/count) blocks, and `effect_size.py` **blocks
acceptance** when the arms span two commits, ran on different accelerators, or
were produced from a tree whose cleanliness could not be verified. Override with
`--allow-mismatched-arms` only when you can explain why the difference is
irrelevant.

Provenance reaches the record through the environment
(`UNIRL_EXPERIMENT_COMMIT` / `_DIRTY` / `_BRANCH`); `run_matrix.py` resolves it
once per matrix and passes it down. The training process does not shell out to
git by default — on this cluster's network work dirs `git rev-parse` costs 6-10s
and `git status` exceeds 30s, and a telemetry field must never delay a launch.
Set `UNIRL_EXPERIMENT_GIT_PROBE=1` to opt into an in-process probe on a fast
filesystem. Unset means `dirty=null`, which is treated as *unverified* and
blocks — never as clean.
