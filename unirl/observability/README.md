# Observability

> **Where it fits:** trainer programs emit run telemetry through this stable
> boundary. Full map: [`../README.md`](../README.md).

`Observer` is the trainer-facing contract for metrics, generated media, console
progress, optimizer-step state, and shutdown. Trainers do not import a concrete
telemetry SDK or adapter. `create_observer(...)` selects the configured provider,
and a disabled observer is the null object used when reporting is off.

The existing configuration remains compatible:

```yaml
logging:
  report_to_wandb: true
  project_name: unirl
  run_name: experiment-1
```

New recipes can name the provider explicitly:

```yaml
logging:
  provider: wandb  # none | wandb
  enabled: true
  project_name: unirl
  run_name: experiment-1
```

`report_to_wandb` remains an alias for selecting and enabling the WandB adapter.
An explicit `provider` takes precedence. Unknown providers fail before training
starts instead of silently dropping telemetry.

Checkpoint `trainer_state.json` files now write provider-neutral
`observer_run_id` and `optimizer_step` fields. The legacy `wandb_run_id` alias is
also written and accepted, so checkpoints resume in either direction across the
migration.

`instrumentation.py` owns phase timing independently of the selected provider.
It wraps the standard rollout, reward, weight-sync, and train collaborators and
injects `perf/<phase>_time_s` at the `Observer.log_rollout_step` boundary.

WandB adaptation, metric extraction, profiling, and driver-side memory
orchestration are owned by this package; worker probes live in the distributed
runtime. See [`MEMORY.md`](MEMORY.md) for the GPU memory workflow.
