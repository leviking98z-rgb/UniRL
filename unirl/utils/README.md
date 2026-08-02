# Utils Compatibility Namespace

`unirl.utils` is no longer an implementation layer. It contains thin,
dependency-lazy compatibility facades for existing imports and dataset-prep
commands. Framework code imports the owning package directly.

| Legacy module | Canonical owner |
|---|---|
| `utils.wandb_logger`, `wandb_metrics` | `observability.wandb`, `observability.metrics` |
| `utils.memory_monitor`, `profiling`, `timing` | `observability.*` |
| `utils.memory_utils` | `distributed.memory` |
| `utils.dtypes`, `hydra` | `config.dtypes`, `config.remote` |
| `utils.distributed_utils`, `graceful_shutdown`, `peft_merge` | `distributed.collectives`, `distributed.process`, `distributed.peft` |
| `utils.adapter_utils` | `models.adapters` |
| `utils.scheduler_utils` | `sde.scheduler` |
| `utils.media`, `shard_balance` | `types.media_conversion`, `types.sharding` |
| `utils.sglang_endpoint` | `rollout.endpoint` |
| `utils.prepare_*` | `data.prepare.*` |
| `utils.misc` | split across `config.imports`, `runtime`, and `observability` |

Compatibility paths remain importable, but new code must use the canonical
owner. `lint/check_utils_ownership.py` enforces that `utils/` stays facade-only
and that framework source does not add new imports from it.
