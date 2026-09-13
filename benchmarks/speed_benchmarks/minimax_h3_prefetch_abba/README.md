# MiniMax-H3 next-batch prompt-prefetch AB/BA harness

This directory contains the **P2 benchmark/protocol harness only**. It does not
start work unless `run_abba.sh` receives an explicit GPU latch, a live
single-node `8×H20` allocation, and a validated P0/P1 integration contract.

The harness is fork-only:

- expected repository: `https://github.com/leviking98z-rgb/UniRL.git`;
- official `origin` push must be disabled;
- no script creates a branch, commit, push, or pull request;
- `protocol_harness.py` is CPU-only and sets `CUDA_VISIBLE_DEVICES=""`.

## Canonical experiment

Four fresh processes run serially on one node:

| Period | Treatment |
|---:|---|
| P1 | prefetch OFF |
| P2 | prefetch ON |
| P3 | prefetch ON |
| P4 | prefetch OFF |

The primary comparisons are `P1/P2` and `P4/P3`; their speedups are combined
with a geometric mean. Every arm fixes:

- `world_size=8`, `sp_size=2`, `dp_size=4`;
- conditioner sharing across each SP2 group;
- eight distinct prompts and four siblings per prompt;
- 32 generated samples and exactly two optimizer updates;
- seed 42 for sampling and data;
- `768×768×124`, 10 denoise steps, `sde_indices=[0,3,6]`, `eta=0.7`;
- `0.5*VideoPickScore(middle)+0.5*CLAP`;
- no persistent prompt-embedding cache and no text-encoder onload;
- one frozen optimizer-step-zero LoRA checkpoint.

The only functional OFF/ON difference is:

```text
bundle.config.prompt_embedding_prefetch=false|true
```

## P0/P1 fail-closed contract

`--p0-contract` must use:

```text
unirl-minimax-h3-p2-integration-contract-v1
```

The contract is intentionally strict. It must bind:

1. **Corrected source ancestry**
   - PR37/P0:
     `b0e7b241f3631bf78f80d9a20ab15413b01b6a79`
   - PR38/P1:
     `9feecf82038f827358ae4686fb8d4045fae7cc72`
   - P2 production:
     `48225e5ec732a3c7a06d2e0cfc61a98286a969cd`
   - the required ancestry is `b0e7b24 -> 9feecf8 -> 48225e5`; each arrow
     means “is the single parent of”;
   - P2 tree:
     `548aaff33347954736d0c086baa1b986f5fcb37c`
   - PR38-to-P2 binary patch SHA-256:
     `f89c68d9996a8d67d16b81bf1daa60795353b5f344a9533df2151d82a3b46d7d`.
2. **P1 source receipt**
   - P1-validated rebased PR38:
     `9feecf82038f827358ae4686fb8d4045fae7cc72`
   - tree:
     `a299ce1d64cf2bde2874f3428e43d5ac4dc3040e`
   - a checksummed
     `unirl-minimax-h3-pr38-p1-source-preflight-v1` receipt for SP2;
   - the sealed P1 archive/full-tree identities;
   - the old `fc2fb26` PR38 head is explicitly rejected.
3. **Sealed P2 source**
   - deterministic archive plus a complete file, symlink, and gitlink
     manifest;
   - exactly the seven P2 production paths;
   - no harness files included in the training source;
   - fork remote
     `https://github.com/leviking98z-rgb/UniRL.git` and disabled official
     push URL.
4. **Group-aware planner**
   - corrected P0 commit `b0e7b241...`;
   - `GroupInterleavedCountPlanner`;
   - implementation digest;
   - a checksummed receipt proving deterministic, disjoint, exhaustive
     four-sibling grouping.
5. **Exactly two updates**
   - two updates indexed `[0,1]`;
   - 16 samples per update, 32 samples total;
   - no omission/duplication;
   - exactly two optimizer steps.
6. **Frozen checkpoint**
   - zero-step LoRA contract, manifest, and receipt;
   - state and optimizer-state digests;
   - verified tensor inventory and empty step-zero optimizer state;
   - a per-arm runtime receipt proving the same checkpoint was loaded before
     the first optimizer step.
7. **P0 runtime overlay and correctness**
   - runtime event schema `unirl-minimax-h3-p0-runtime-event-v2`;
   - reviewed P0 overlay SHA-256
     `557c4ece42f2b3200577c6ee6aeaa387eaadcf36a2be8d381dfd1e4746714388`;
   - prompt manifest;
   - sample-level reward/component/output fingerprints;
   - update membership;
   - in-process hot-cache contract with zero disk activity and no fallback.

Any missing field, wrong schema, digest mismatch, stale PR38/P2 head, changed
source, malformed membership, or checkpoint mismatch aborts before a result is
accepted.

## Overlap measurement

The end-to-end metrics are:

```text
perf/step_time_s
perf/generate_time_s
perf/reward_time_s
perf/train_time_s
```

The direct overlap evidence comes from MiniMax-H3 embedding telemetry. With
SP2 sharing there are 64 rows per arm: 32 source-rank rows and 32 receiver-rank
rows.

Expected source-rank pattern:

```text
OFF: 8 synchronous misses + 24 memory hits
ON:  4 synchronous misses + 28 memory hits
```

Every receiver row must be one shared hit with zero local memory/disk/miss
activity. An ON arm that still has eight source misses, reports fallback, uses
the persistent disk cache, or emits a digest/token mismatch is failed rather
than counted as a speedup.

## Correctness measurement

The summarizer accepts performance only when all arms have:

- identical functional configs apart from the prefetch flag;
- identical prompts, seeds, frozen checkpoint, planner receipt, and source;
- exactly 32 unique sample ledger rows;
- finite total/component rewards;
- identical per-sample correctness fingerprints across OFF and ON;
- the canonical two-update group-interleaved partition;
- exactly two finite, nonzero gradient norms/optimizer updates;
- complete NVML coverage for GPUs 0–7;
- no traceback, OOM, NCCL failure, nonfinite metric, or prefetch fallback.

## CPU-only validation

Run from this directory:

```bash
CUDA_VISIBLE_DEVICES="" python3 protocol_harness.py
ruff check .
ruff format --check .
PYTHONDONTWRITEBYTECODE=1 python3 -m compileall -q .
bash -n launch_arm.sh run_abba.sh
```

`protocol_harness.py` builds synthetic contracts and artifacts. It covers a
golden PASS plus fail-closed checks for bounded prefetch, producer exceptions,
shutdown/cancellation/join, corrected source ancestry and gitlinks, AB/BA
pairing, planner membership, frozen checkpoint, event sidecars, overlap,
cleanup, prompt/output fingerprints, nonfinite metrics, disk-cache activity,
and event tampering. It loads `prefetch.py` directly and never imports or
initializes CUDA.

Seal the immutable production commit separately from the later harness commit:

```bash
python3 seal_source.py \
  --repo /root/megascale_p2_prefetch \
  --output-dir /root/p2_cpu_tmp/p2-source-seal-48225e5
```

The current deterministic seal for `48225e5` is:

```text
archive SHA-256:   1738ba4ecadfdc076b479cd560aef9bd1ed454972ca1e213c9a4abf0b2d9d40d
manifest SHA-256:  9196d005c370120d46c3933a78dca5e1576b854dced256d22847ee57c8308e0b
full-tree SHA-256: 8dc990be142da07e1a56c6b421ac398becdb16daff0ba286fe99030a3e30b799
files/symlinks/gitlinks: 1056/0/1
```

## Reviewed launch shape — do not run before the P0 gate

Launch is forbidden until P0/P1 explicitly select SP2 and a reviewer supplies
the integration contract that binds the source seal, reviewed P0 v2 runtime
overlay, prompt manifest, and frozen zero-step LoRA checkpoint:

```bash
P2_PREFETCH_CONFIRM_START=YES \
P2_PREFETCH_P0_CONTRACT=/root/shared/.../p2_integration_contract_v1.json \
P2_PREFETCH_ALLOCATION=alloc_xxx \
P2_PREFETCH_CAMPAIGN_ID=p2-prefetch-abba-YYYYMMDDTHHMMSSZ \
bash benchmarks/speed_benchmarks/minimax_h3_prefetch_abba/run_abba.sh
```

Expected wall time is roughly **2–3 hours** for four fresh-process arms,
depending mainly on model load, video generation, reward scoring, and
checkpoint I/O. This repository snapshot does not execute that command.

## Failure cleanup and partial evidence

`run_abba.sh` installs `EXIT/ERR/INT/TERM` handling. On failure it:

1. stops only the current campaign's NVML sampler;
2. verifies the current workload PID/starttime/token before signaling its
   process group;
3. copies any available current-arm evidence to `partial-remote/`;
4. writes atomic `partial-summary.json`;
5. runs the documented node cleanup on the allocated node.

The trap never releases the allocation and never touches another campaign by
PID alone. A partial summary always has `completed=false` and is not usable as
a performance result.

For a reviewed allocation, the snapshot/cleanup failure path can be exercised
without starting training by adding:

```bash
P2_PREFETCH_DRY_RUN=1 P2_PREFETCH_TEST_FAIL_AFTER_SNAPSHOT=1
```

The injected failure must produce an atomic `partial-summary.json` with
`completed=false` and `performance_usable=false`.
