# MiniMax-H3 workload trace and grouped-reordering simulator

This CPU-only tool measures the scheduling information available around the native
MiniMax-H3 trainside pipeline and estimates the best-case value of grouped
reordering before changing the distributed scheduler.

## Current data path and limitation

`TextPromptDataset` normalizes a prompt plus optional metadata/media references.
`MultimodalRLDataSource` collates those rows into a text-rooted `Sample`, and
`DiffusionTrainer._build_request_sample` forks each root by
`sampling.samples_per_prompt`. The generation `Part` carries one shared
`DiffusionSamplingParams`; `MiniMaxH3Pipeline.generate` resolves
`height`, `width`, and `num_frames` from that shared object.

Consequences:

- prompt token length varies per root and is known only after the Qwen3-VL
  conditioner runs;
- output video geometry is not a dataset field and is fixed within one run;
- source video dimensions/frame counts in video datasets do not override the H3
  output geometry;
- `LatentSegment` stores dense concatenated tensors, so different latent shapes
  cannot share one current rollout batch;
- DP scatter assigns complete root trees. All `samples_per_prompt` siblings of a
  root are therefore an indivisible scheduling unit.

The canonical trainside recipe pins `768x768x124` and
`rollout.forward_batch_size=1`. The NFT recipe pins `256x384x124`; it is another
fixed-geometry run, not a mixed workload. To study geometry skew today, profile
separate fixed-geometry runs and pass all of their traces to `analyze`. A real
mixed-geometry GPU A/B still needs request-level sampling params plus a
shape-aware scheduler/worker pool.

## Trace collection

Tracing is disabled by default. Add one field under `bundle.config`:

```yaml
bundle:
  config:
    workload_telemetry_path: /shared/traces/h3-768x768x124.jsonl
```

Only SP rank zero writes, so one generated sample produces one record rather
than one record per SP worker. Writers use an advisory file lock. Each row
contains sample/root IDs, canvas and frame count, text/video/audio/packed row
counts, SP padding and placement, plus synchronized text-embedding, denoising,
decode, and total wall times. Enabling the trace introduces device
synchronizations and is intended for bounded profiling runs, not normal
training. With the field absent or `null`, no clocks, synchronization, or file
I/O run.

## Fail-closed fixed-geometry matrix

`matrix_driver.py` binds the four profiles to one clean source commit/tree, the
canonical and resolved Hydra config, prompt bytes and deterministic root IDs,
the pretrained-model locator, one completed adapter checkpoint and its LoRA
metadata, and the Python executable. The output directory must be outside the
source checkout. Any later source/config/prompt/checkpoint mutation is rejected.

The intended two-node profile is 16 GPUs with SP8, yielding two DP groups and a
local reorder group of two. `num_prompts` must also satisfy the colocated reward
DP divisibility check; 16 is the minimum for the canonical recipe.

```bash
export PRETRAINED_MODEL=/shared/models/MiniMax-Hailuo-2.3

python benchmarks/video/minimax_h3/matrix_driver.py prepare \
  --prompts /shared/p3/prompts.jsonl \
  --lora-checkpoint /shared/p3/checkpoint-N \
  --output-dir /shared/p3/fixed-matrix \
  --num-devices 16 \
  --sp-size 8 \
  --group-size 2 \
  --num-prompts 16 \
  --cost total_s
```

Preparation prints four `run-one` commands. Review each exact train command
without launching it:

```bash
RAY_ADDRESS=auto python benchmarks/video/minimax_h3/matrix_driver.py show \
  --manifest /shared/p3/fixed-matrix/matrix.json \
  --geometry g0
```

The driver calls `python -m unirl.train_diffusion` directly and requires an
already-running P0 Ray allocation; it does not call a launcher, `ray start`, or
`ray stop`. GPU execution is additionally locked unless `P3_ALLOW_GPU_RUN=1` is
set. Run all four printed commands only after P0:

```bash
export RAY_ADDRESS=auto
export P3_ALLOW_GPU_RUN=1
python benchmarks/video/minimax_h3/matrix_driver.py run-one \
  --manifest /shared/p3/fixed-matrix/matrix.json \
  --geometry g0
```

The loaded LoRA is frozen for this collection. The checkpoint step becomes
`num_rollouts`, so startup evaluation runs at that step and the training loop is
empty: no optimizer update occurs. Each successful row writes a trace plus a
receipt binding its digest and exact command.

Analyze only after all four receipts exist:

```bash
python benchmarks/video/minimax_h3/matrix_analyzer.py \
  --manifest /shared/p3/fixed-matrix/matrix.json \
  --output /shared/p3/fixed-matrix/analysis.json \
  --require-go
```

The analyzer validates every binding again, reconstructs deterministic order
inside each fixed trace by DP rank, then round-robin merges G0-G3 into the
predeclared mixed workload. Exit code 2 means invalid or changed evidence. With
`--require-go`, exit code 3 means a valid `NO-GO`.

### Mixed geometry and prompt-length replay

`mixed_trace_analyzer.py` adds two stricter workload modes:

1. `fixed-replay` composes counterfactual mixed waves from the four validated
   fixed traces. Every wave contains the same number of G0-G3 roots, and every
   root sees every geometry once per four-wave cycle. Multiple deterministic
   schedule trials expose whether a result depends on a lucky input order.
2. `mixed` consumes already-mixed JSONL waves. It requires the same bound root
   and sibling set, legal G0-G3 geometry, one geometry per root, the declared SP
   topology, equal recorded DP capacity, and prompt token counts matching the
   fixed traces. The recorded DP ranks are the baseline assignment.

Both modes schedule complete root trees, never individual siblings. Structural
predictors include packed rows, SP-padded rows, and squared SP-padded rows.
When the fixed profiles are measured GPU traces, `fixed-replay` evaluates those
assignments against measured `total_s`; with CPU proxy traces it stays explicitly
in the structural proxy domain. Text length is real only when the conditioner
trace reports nonzero `text_tokens`.

```bash
python benchmarks/video/minimax_h3/mixed_trace_analyzer.py fixed-replay \
  --manifest /shared/p3/fixed-matrix/matrix.json \
  --trials 64 \
  --waves 4 \
  --output /shared/p3/fixed-matrix/mixed-replay.json \
  --plan-output /shared/p3/fixed-matrix/grouped-ab-plan.json

python benchmarks/video/minimax_h3/mixed_trace_analyzer.py mixed \
  --manifest /shared/p3/fixed-matrix/matrix.json \
  --trace /shared/p3/mixed/wave-0.jsonl \
  --trace /shared/p3/mixed/wave-1.jsonl \
  --predictor-cost attention_rows2 \
  --outcome-cost total_s \
  --require-mixed-length \
  --output /shared/p3/mixed/analysis.json \
  --plan-output /shared/p3/mixed/grouped-ab-plan.json
```

The fixed-replay implementation gate uses the p10 tail ratio and p10 predicted
speedup across schedule trials. A `GO` therefore cannot be produced by one
fortunate permutation. If any declared predictor fails the 5% gate, or the
predictors disagree across it, the overall proxy result is `NO-GO`. This is
deliberately more conservative than the single round-robin matrix summary.

### Auditable CPU substitute

When GPU collection is prohibited, prepare the same matrix with
`--cost packed_rows`, `padded_rows`, or `attention_rows2`, then materialize the
declared geometry-only proxy:

```bash
python benchmarks/video/minimax_h3/matrix_proxy.py \
  --manifest /shared/p3/fixed-matrix/matrix.json
python benchmarks/video/minimax_h3/matrix_analyzer.py \
  --manifest /shared/p3/fixed-matrix/matrix.json \
  --output /shared/p3/fixed-matrix/proxy-analysis.json
```

Proxy receipts are explicitly marked `analytical_cpu_proxy`: text tokens and
all timings are fixed to zero. They validate the matrix, load model, and
decision logic, but cannot support a measured performance or end-to-end ROI
claim. `matrix_proxy.py` rejects timing costs such as `total_s` and `denoise_s`;
the analyzer rejects mixed or mislabeled evidence.

## Paired grouped-reordering A/B contract

`grouped_reordering_ab.py` is a CPU-only harness for preparing and analyzing the
future GPU experiment. It never launches Ray, a trainer, or a GPU process. It
content-addresses the source matrix, trace files, mixed workload, per-rank
assignments, topology, and success thresholds, then emits one result template
per arm:

```bash
python benchmarks/video/minimax_h3/grouped_reordering_ab.py prepare \
  --plan /shared/p3/fixed-matrix/grouped-ab-plan.json \
  --output /shared/p3/fixed-matrix/grouped-ab-contract.json \
  --templates-dir /shared/p3/fixed-matrix/result-templates \
  --measurement generation_s \
  --repetitions 3 \
  --warmup-repetitions 1

python benchmarks/video/minimax_h3/grouped_reordering_ab.py validate \
  --contract /shared/p3/fixed-matrix/grouped-ab-contract.json

python benchmarks/video/minimax_h3/grouped_reordering_ab.py analyze \
  --contract /shared/p3/fixed-matrix/grouped-ab-contract.json \
  --baseline /shared/p3/results/baseline.json \
  --grouped /shared/p3/results/grouped_lpt.json \
  --output /shared/p3/results/paired-analysis.json \
  --require-go
```

The result analyzer rejects a changed workload digest, sample set, rank
assignment, topology, incomplete sample count, invalid per-rank timing, or
different output digest. The default measured `GO` requires all of:

- aggregate and median paired speedup at least 1.05x;
- treatment faster in at least 75% of paired waves;
- absolute paired reward-mean delta at most 0.01;
- tracing overhead at most 0.5%;
- identical order-independent output digests.

The generated plan currently says
`directly_executable_on_current_unirl: false`. It is an exact workload and result
contract, not a hidden GPU launcher. Current H3 rollout has one shared
`DiffusionSamplingParams` and one dense latent shape per `Sample`; executing the
mixed plan requires the small scheduler/runtime extension described below.

## Simulator

`--ranks` always means **DP groups**, not physical GPUs. For eight physical GPUs:
SP1 has 8 DP groups, SP2 has 4, SP4 has 2, and SP8 has 1. `--group-size` is the
number of adjacent DP groups allowed to exchange roots; use `1` as the no-op
control or `--ranks` for a global upper bound.

```bash
python benchmarks/video/minimax_h3/grouped_reordering.py analyze \
  /shared/traces/h3-*.jsonl \
  --ranks 8 \
  --group-size 4 \
  --cost packed_rows \
  --output /tmp/h3-grouped-summary.json
```

The baseline is equal-count contiguous dispatch. The alternative is stable,
equal-capacity longest-processing-time-first assignment inside each rank group.
Both keep every root's siblings together and keep the same number of roots on
each rank. Reported metrics include rank loads, `max/mean` tail ratio,
`(max-min)/mean` imbalance, efficiency, assignment/permutation, and predicted
makespan speedup.

With multiple input traces, roots are round-robin interleaved by default. This
avoids treating four fixed-geometry profile files as four artificial
geometry-sorted blocks. Use `--merge-order input` only when file concatenation
is the intended arrival order. The fixed-matrix analyzer additionally uses
DP-rank source order because concurrent JSONL appends do not preserve stable
cross-rank arrival order.

Cost choices:

- `packed_rows`: linear unpadded token-work proxy;
- `padded_rows`: linear token-work proxy after SP divisibility padding;
- `attention_rows2`: quadratic attention proxy;
- `denoise_s`: measured denoising wall time;
- `total_s`: measured complete generation wall time.

A deterministic synthetic trace is useful for CPU validation only:

```bash
python benchmarks/video/minimax_h3/grouped_reordering.py synthesize \
  --geometry 768x768x124:8 \
  --geometry 768x1024x124:8 \
  --geometry 768x768x175:8 \
  --geometry 768x1024x175:8 \
  --text-tokens 64,128,256 \
  --samples-per-prompt 4 \
  --sp-size 4 \
  --output /tmp/h3-mixed.jsonl
```

Synthetic row proxies are not performance claims. Prefer `denoise_s` or
`total_s` from real fixed-geometry profiles when deciding whether scheduler work
is justified.

## Follow-up GPU A/B matrix

Start from `minimax_h3_t2va_trainside.yaml`. Create four temporary runtime
variants that change only these fields and write separate traces:

| Profile | `sampling.height` | `sampling.width` | `sampling.num_frames` | Base packed rows, excluding text | Relative to canonical |
|---|---:|---:|---:|---:|---:|
| G0 | 768 | 768 | 124 | 21,726 | 1.00x |
| G1 | 768 | 1024 | 124 | 28,830 | 1.33x |
| G2 | 768 | 768 | 175 | 30,536 | 1.41x |
| G3 | 768 | 1024 | 175 | 40,520 | 1.86x |

For each geometry, preserve the same prompt IDs, seeds, denoise steps,
`samples_per_prompt`, model/checkpoint, and reward settings. Evaluate topology
rows separately because the simulator rank count is DP groups:

| Physical GPUs | SP | DP groups passed as `--ranks` | Suggested local `--group-size` sweep |
|---:|---:|---:|---|
| 16 | 1 | 16 | 1, 4, 8, 16 |
| 16 | 2 | 8 | 1, 4, 8 |
| 16 | 4 | 4 | 1, 2, 4 |
| 16 | 8 | 2 | 1, 2 |
| 32 | 1 | 32 | 1, 4, 8, 16, 32 |
| 32 | 2 | 16 | 1, 4, 8, 16 |
| 32 | 4 | 8 | 1, 4, 8 |
| 32 | 8 | 4 | 1, 2, 4 |
| 128 | 1 | 128 | 1, 8, 16, 32 |
| 128 | 2 | 64 | 1, 8, 16, 32 |
| 128 | 4 | 32 | 1, 4, 8, 16, 32 |
| 128 | 8 | 16 | 1, 4, 8, 16 |

The current code can collect those four profiles and simulate their combined
multiset. It cannot execute the combined multiset in one training run yet.

## Go/no-go criteria

Stop before scheduler implementation if either condition holds:

- contiguous `tail_ratio < 1.05`; or
- grouped LPT predicts less than 5% makespan improvement.

For analytical evidence, report the result as `proxy GO` or `proxy NO-GO`;
measured GO/NO-GO remains pending until the four real fixed-geometry traces are
collected.

Before trusting a later GPU result, require:

1. tracing-on overhead below 0.5% on the same fixed workload;
2. identical prompt/seed/geometry multisets and unchanged outputs/reward within
   the established baseline tolerance;
3. at least 5% paired generation or end-to-end step throughput improvement;
4. the improvement direction repeats across multiple post-warmup waves;
5. no root siblings cross DP groups and no rank receives a different root count.

## Minimum real-GPU implementation and experiment

The minimum runtime change is intentionally narrower than a new batching
system:

1. Carry a per-root `DiffusionSamplingParams` (or an equivalent immutable
   geometry key) from request construction to trainside dispatch.
2. Before `DP_SCATTER`, group complete root trees by a deterministic structural
   estimate and apply the plan's equal-capacity permutation inside each allowed
   DP-rank group. Never split `samples_per_prompt` siblings.
3. Keep `rollout.forward_batch_size=1`; each DP group may execute different
   shapes sequentially, so no heterogeneous dense latent tensor is required.
4. Record DP-group start/end events around the whole generation wave, plus
   successful sample count, reward mean, and an order-independent digest of
   sorted `sample_id -> generated artifact` hashes.
5. Run one trace-off/trace-on control to measure telemetry overhead, one warmup
   repetition per arm, then at least three measured paired repetitions. Alternate
   arm order by repetition to reduce thermal/order bias.

For the existing 16-GPU/SP8 topology this is two DP groups, four mixed waves,
and 12 measured waves per arm after warmup. The baseline and treatment use the
same frozen LoRA, prompts, per-sample seeds/noise, geometry assignment, reward
configuration, and sample count; only the DP-root permutation differs.

## CPU tests

```bash
PYTHONPATH=. python -m unittest -v benchmarks/video/minimax_h3/test_p3_cpu.py
```

The tests cover balanced geometry crossover, nonuniform prompt lengths,
recorded DP placement, plan semantic regeneration, A/B GO and NO-GO branches,
and rank-assignment tamper rejection.
