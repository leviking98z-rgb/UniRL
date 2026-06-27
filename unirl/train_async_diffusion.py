#!/usr/bin/env python
"""UniRL async diffusion training entry point (Hydra-native).

Sibling of ``train_diffusion.py`` that drives
:class:`unirl.trainer.async_diffusion.AsyncDiffusionTrainer` — the disaggregated,
async variant of the diffusion path (training and a resident dedicated-rollout
engine on DISJOINT GPU slabs, generation overlapped with training, weights pushed
cross-slab via ``NCCLWeightSync``). The synchronous colocate/separate trainer is
unchanged; this is purely additive. The trainer forces ``layout="separate"``.

Launch (single node):
  python -m unirl.train_async_diffusion \
      --config-name diffusion/hunyuan_video15/hunyuan_video15_t2v_async num_devices=8

Extra config knobs vs the synchronous separate recipe:
  * ``max_inflight`` — concurrent generations (overlap depth). ``1`` ≈ one-step pipeline.
  * ``buffer_max_staleness`` — weight-syncs a buffered group may cross before eviction.
    ``0``/unset = on-policy (the launch clamp never lets a generation cross a sync);
    ``>0`` = bounded off-policy continuous buffer (TCSS).
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.async_diffusion import AsyncDiffusionTrainer


@hydra.main(version_base=None, config_path="../examples",
            config_name="diffusion/hunyuan_video15/hunyuan_video15_t2v_async")
def main(cfg: DictConfig) -> None:
    trainer = AsyncDiffusionTrainer(
        # ---- async knobs ----
        max_inflight=int(cfg.get("max_inflight", 1)),
        buffer_max_staleness=cfg.get("buffer_max_staleness"),
        # ---- DiffusionTrainer kwargs (layout forced to "separate" by the trainer) ----
        cfg=cfg,
        batch_size=cfg.batch_size,
        bundle_cfg=cfg.bundle,
        pipeline_cfg=cfg.pipeline,
        backend_cfg=cfg.backend,
        rollout_cfg=cfg.rollout,
        reward_cfg=cfg.reward,
        algorithm_cfg=cfg.algorithm,
        stack_cfg=cfg.stack,
        data_source_cfg=cfg.data_source,
        sampling_cfg=cfg.sampling,
        sync_cfg=cfg.get("sync"),
        logging_cfg=cfg.get("logging"),
        train_fraction=cfg.get("train_fraction", 0.5),
        adv_use_global_std=cfg.get("adv_use_global_std", False),
        eval_interval=cfg.get("eval_interval", 0),
        eval_num_prompts=cfg.get("eval_num_prompts", 64),
        eval_samples_per_prompt=cfg.get("eval_samples_per_prompt", 4),
        eval_chunk_prompts=cfg.get("eval_chunk_prompts", 16),
        eval_cfg_text_scale=cfg.get("eval_cfg_text_scale", 4.0),
        eval_eta=cfg.get("eval_eta", 0.0),
        stage_config=cfg.get("stage_config"),
    )
    trainer.train(
        num_rollouts=cfg.get("num_rollouts", 100),
        weight_sync_interval=cfg.get("weight_sync_interval", 1),
        save_interval=cfg.get("save_interval", 0),
        save_dir=cfg.get("save_dir"),
        load_dir=cfg.get("load_dir"),
        save_mode=cfg.get("save_mode", "auto"),
    )


if __name__ == "__main__":
    main()
