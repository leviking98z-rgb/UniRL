# Config

> **Where it fits:** cross-cutting — not a box in the loop. Every box (rollout,
> reward, train, sync) is built from a config dataclass whose field checks and
> precision aliases this module provides. Full map: [`../README.md`](../README.md).

## What it is

`unirl.config` is the shared toolkit behind UniRL's flat-recipe config flow.
Component-specific dataclasses still live next to the components that consume
them. This package owns shared field validators and the typed driver-side
execution plan that composes those components.

## Why it exists

A recipe is one flat YAML wired entirely by `_target_` dotpaths — there are **no**
Hydra config groups and no `defaults:` lists. That keeps every run reproducible
from a single file, but it also means Hydra type-checks nothing. This module is
where invariants get enforced instead:

- Each dataclass fails fast in `__post_init__` via `require(...)`, with a clear
  `ValueError`.
- Every precision field accepts the same aliases (`bf16`/`bfloat16`, `fp16`/…,
  `fp32`/…) through one shared `validate_precision_type`, so the rules and error
  message are identical everywhere.
- Rollout engines and weight-sync implementations declare
  `CAPABILITIES: ComponentCapabilities` on their own classes. The declaration
  describes direct/dedicated ownership, generation shape, lifecycle and
  weight-receiver/transport support.
- `ExecutionPlan.from_config` resolves those declarations into a
  `CapabilityGraph`, normalizes single- and multi-track engine/sync selections,
  computes role placement, and validates engine ↔ sync, loop, layout and offload
  compatibility before any GPU actor is created.

## How it works

A recipe is one flat YAML marked `# @package _global_`. Components are `_target_`
dotpaths, sub-configs are nested `_target_` blocks, shared values are `${...}`
interpolations. There is no ConfigStore and no registration step.

Instantiation is a **driver-routes / worker-materializes** split:

- `parse_hydra_cfg` (`../utils/hydra.py`) resolves only the *top-level* `_target_`
  on the driver and passes nested blocks through as plain dicts.
- `Worker._resolve_init_kwargs` (`../distributed/group/worker.py`) walks the tree on
  the worker and builds each nested `_target_` with `get_method(_target_)(**children)`
  — deliberately **not** `hydra.utils.instantiate`, so already-built objects pass
  through unchanged and each is constructed in the worker's own CUDA context.

Validation runs in three layers:

- **Per-dataclass `__post_init__`** — local field invariants via `require(...)`
  and `validate_precision_type(...)` (at actor-build time).
- **Typed execution planning** — `BaseTrainer.__init__` creates
  `self.execution_plan` before `DevicePool`. Trainer subclasses declare only
  their `LOOP_KIND` and an optional fixed `PLACEMENT_OVERRIDE`; component
  compatibility comes from the graph, not constructor reflection or `_target_`
  string matching.
- **Cross-component validators** (`validate_weight_sync_contract`,
  `validate_rollout_layout`, `validate_offload_contract`, …) remain available to
  older config assembly paths and consult the same class-owned capabilities.

**Extending it:** a new component config is a plain `@dataclass` next to the
component (not here), with `require(...)` checks in `__post_init__`. A new rollout
engine or sync implementation must declare `CAPABILITIES` on the concrete class;
the CPU framework-contract guard enforces that declaration. Add a capability only
when it represents a peer-composition decision, then validate it in
`ExecutionPlan` rather than adding an engine-name table.

## Gotchas

- `ExecutionPlan` resolves only top-level component classes. It does not
  instantiate nested model/runtime configs; those still materialize on workers.
- Multi-track composition accepts either `rollout` or `ar_rollout` +
  `dit_rollout`, and either one shared `sync` or track-keyed `sync.ar` /
  `sync.diffusion`. A dedicated engine must have a compatible sync path.
- **`# @package _global_` on line 1 is mandatory** — omit it and Hydra nests the
  whole recipe under a bucket key, so `cfg.batch_size` won't resolve.
- **`validate_precision_type` validates but does not normalize** — it *returns* the
  canonical alias (`bf16`), but every call site invokes it as a bare statement and
  discards the result. So `model_precision: bfloat16` stays the raw string in `cfg`;
  downstream code must re-parse it with `parse_torch_dtype` itself.
