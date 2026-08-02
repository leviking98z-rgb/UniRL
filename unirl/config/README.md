# Config

> **Where it fits:** cross-cutting — not a box in the loop. Every box (rollout,
> reward, train, sync) is built from a config dataclass whose field checks and
> precision aliases this module provides. Full map: [`../README.md`](../README.md).

## What it is

`unirl.config` is the shared toolkit behind UniRL's bounded-composition recipe
flow.
Component-specific dataclasses still live next to the components that consume
them. This package owns shared field validators and the typed driver-side
execution plan that composes those components.

## Why it exists

A public recipe is one stable entry YAML wired entirely by `_target_` dotpaths.
It may inherit common fields from exactly one private `examples/_base/` file;
the public file then overlays the choices that identify that experiment. Bases
cannot inherit from other bases, so resolving a run never requires following a
deep config-group graph. Hydra still type-checks none of these plain mappings.
This module is where invariants get enforced instead:

- Each dataclass fails fast in `__post_init__` via `require(...)`, with a clear
  `ValueError`.
- Every precision field accepts the same aliases (`bf16`/`bfloat16`, `fp16`/…,
  `fp32`/…) through one shared `validate_precision_type`, so the rules and error
  message are identical everywhere.
- Rollout engines and weight-sync implementations declare
  `CAPABILITIES: ComponentCapabilities` on their own classes. The declaration
  describes direct/dedicated ownership, generation shape, lifecycle and
  weight-receiver/transport support.
- `rollout.py` defines dependency-light structural protocols for memory
  lifecycle and tensor/NCCL/IPC/LoRA/checkpoint receivers. Capability resolution
  verifies that every declared rollout capability has its required method
  surface; the stdlib framework guard performs the same check without importing
  GPU dependencies.
- `ExecutionPlan.from_config` resolves those declarations into a
  `CapabilityGraph`, normalizes single- and multi-track engine/sync selections,
  computes role placement, and validates engine ↔ sync, loop, layout and offload
  compatibility before any GPU actor is created.
- `ModelPluginPlan.from_config` resolves package-local model manifests and
  validates bundle ↔ pipeline ↔ config ↔ backend ↔ algorithm ↔ rollout stage
  compatibility without importing heavyweight bundle modules.

## How it works

A recipe and its optional one-layer base are marked `# @package _global_`.
Components are `_target_` dotpaths, sub-configs are nested `_target_` blocks,
and shared runtime values are `${...}` interpolations. There is no ConfigStore
and no registration step. `lint/check_recipe_composition.py` keeps composition
bounded, checks references and cycles, and composes every public entry.

Instantiation is a **driver-routes / worker-materializes** split:

- `parse_hydra_cfg` (`remote.py`) resolves only the *top-level* `_target_`
  on the driver and passes nested blocks through as plain dicts.
- `Worker._resolve_init_kwargs` (`../distributed/group/worker.py`) walks the tree on
  the worker and builds each nested `_target_` with `get_method(_target_)(**children)`
  — deliberately **not** `hydra.utils.instantiate`, so already-built objects pass
  through unchanged and each is constructed in the worker's own CUDA context.

Validation runs in three layers:

- **Per-dataclass `__post_init__`** — local field invariants via `require(...)`
  and `validate_precision_type(...)` (at actor-build time).
- **Typed execution planning** — `BaseTrainer.__init__` creates
  `self.model_plan` and `self.execution_plan` before `DevicePool`. The model plan
  uses package-local `ModelPluginSpec` declarations; the execution plan uses the
  capability graph. Trainer subclasses declare only their `LOOP_KIND` and an
  optional fixed `PLACEMENT_OVERRIDE`.
- **Cross-component validators** (`validate_weight_sync_contract`,
  `validate_rollout_layout`, `validate_offload_contract`, …) remain available to
  older config assembly paths and consult the same class-owned capabilities.

**Extending it:** a new component config is a plain `@dataclass` next to the
component (not here), with `require(...)` checks in `__post_init__`. A new rollout
engine or sync implementation must declare `CAPABILITIES` on the concrete class;
the CPU framework-contract guard enforces both the declaration and its structural
method surface. Add a capability only when it represents a peer-composition
decision, then validate it in `ExecutionPlan` rather than adding an engine-name
table.

## Gotchas

- Driver-side plans validate composition but do not instantiate nested model or
  runtime configs; those still materialize on workers.
- Multi-track composition accepts either `rollout` or `ar_rollout` +
  `dit_rollout`, and either one shared `sync` or track-keyed `sync.ar` /
  `sync.diffusion`. A dedicated engine must have a compatible sync path.
- **`# @package _global_` on line 1 is mandatory** — omit it and Hydra nests the
  whole recipe under a bucket key, so `cfg.batch_size` won't resolve.
- Public entries may reference one private base and must merge `_self_` last;
  bases cannot contain `defaults:`. This is an intentional reuse boundary, not a
  general Hydra inheritance system.
- **`validate_precision_type` validates but does not normalize** — it *returns* the
  canonical alias (`bf16`), but every call site invokes it as a bare statement and
  discards the result. So `model_precision: bfloat16` stays the raw string in `cfg`;
  downstream code must re-parse it with `parse_torch_dtype` itself.
