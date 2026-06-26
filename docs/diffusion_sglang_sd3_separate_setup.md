# SD3 sglang rollout (colocate & separate) — setup & gotchas

Reproducible bring-up notes for running `examples/diffusion/sd3_sglang_rollout_colocate.yaml`
and `examples/diffusion/sd3_sglang_lora_separate.yaml` on a cuda-compat-13 / driver-535
cluster. All items below were hit + resolved on real 32-GPU multinode runs (SD3.5-medium,
PickScore reward); both layouts train correctly (reward 0.77 → 0.85).

## 1. ★ Root cause to know first: the model path **name** selects the pipeline config

sglang-diffusion resolves the *pipeline class* from `model_index.json` (`_class_name`), but
resolves the *pipeline config* (`StableDiffusion3PipelineConfig`) via a **substring match on
the `model_path` string** (`registry.py::register_configs(model_detectors=[...])`, e.g.
`"stable-diffusion-3.5-medium" in path`).

If `model_path` does **not** contain a registered substring (e.g. a renamed local checkpoint
dir like `/data/sd35-clean`), config resolution silently falls back to the **base
`PipelineConfig`**, which cascades into a pile of confusing downstream failures:

- `ValueError: Duplicate tensor names detected` (the SD3 config's safetensors variant filter
  never runs, so both `model.safetensors` + `model.fp16.safetensors` get loaded),
- `Running grouped pipeline stages: []` → empty pipeline → no forward,
- `IndexError: tuple index out of range` at `text_encoder_loader.py` (`text_encoder_precisions`
  is empty on the base config),
- `ZeroDivisionError` in `gpu_worker.py::do_mem_analysis` (0 reserved bytes ⇒ divide-by-zero).

**Fix:** make `PRETRAINED_MODEL` (model_path) a directory whose name contains a registered
substring, e.g. symlink:

```bash
ln -s /path/to/cleaned-sd35 /path/to/stable-diffusion-3.5-medium
export PRETRAINED_MODEL=/path/to/stable-diffusion-3.5-medium
```

Do **not** rename the checkpoint to an arbitrary name — the detector substring is load-bearing.

## 2. Environment bring-up (cuda-compat-13 / driver-535), per node

- **compat libcuda must be in `ldconfig`** (Ray strips `LD_LIBRARY_PATH`, so the Ray workers
  only find the cuda-13 compat `libcuda.so.1` via the system cache):
  ```bash
  echo /opt/cudacompat13/usr/local/cuda-13.3/compat > /etc/ld.so.conf.d/000_cudacompat13.conf
  ldconfig
  ```
  Missing this ⇒ `ValueError: ProcessGroupNCCL is only supported with GPUs, no GPUs found!`
  in `Worker.setup_global_pg()`. **A stale Ray cluster will not pick up a fresh `ldconfig`** —
  fully tear Ray down (`node_cleanup`) and start fresh after editing the cache.

- **cuDNN-SDPA aborts CLIP attention/conv** (PickScore reward + diffusion text encoders) with
  a native SIGABRT (no Python traceback). Opt in to the shim:
  ```bash
  export UNIRL_DISABLE_CUDNN=1   # routes SDPA/conv through native CUDA kernels process-wide
  ```
  Implemented in `unirl/__init__.py::_maybe_disable_cudnn()` (every Ray actor imports `unirl`).

## 3. Upstream sglang-diffusion bug (defensive)

`gpu_worker.py::do_mem_analysis` computes `pool_overhead_gb / peak_reserved_gb` for a debug
log line with no zero-guard; if a forward reserved 0 bytes this raises `ZeroDivisionError`.
With item (1) fixed this no longer triggers (forwards actually run), but if patching the
pinned `sglang[diffusion]==0.5.12.post1` venv directly, guard it:
`(pool_overhead_gb / peak_reserved_gb * 100 if peak_reserved_gb else 0.0)`.

## 4. Observed: separate is ~3× faster than colocate for diffusion RL

Same 32 GPU / SD3 / 30 denoise steps / same env, only `layout` differs:

| layout | steady step time | mechanism |
|---|---|---|
| `separate` (train/rollout split slabs) | **~50 s/step** | sglang rollout stays resident on its slab |
| `colocate` (time-share, sglang sleep/wake) | ~156 s/step | every step does full `release_memory_occupation` (release + realloc engine memory) |

For diffusion RL the colocate cost is **not** rollout/train bubble (bubble ≈ 0) but the
**per-step memory sleep/wake tax** of time-sharing the GPUs; `separate` trades extra GPUs for
keeping the rollout engine resident and wins ~3×. (Opposite of the usual text-LLM intuition
where colocate is preferred.)
