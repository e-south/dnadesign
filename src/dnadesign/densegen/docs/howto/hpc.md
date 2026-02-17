## DenseGen HPC runbook

This how-to guide provides scheduler-safe DenseGen execution patterns for batch environments. Read it when you need predictable preflight checks, resume behavior, and artifact generation on shared clusters.

### When to use this runbook
This section clarifies scope so operators can quickly choose the right guide.

Use this runbook for generic scheduler workflows; use **[BU SCC delta](bu-scc.md)** for BU-specific flags and policy details.

### Preflight checks
This section runs the minimum assertions before submitting long jobs.

```bash
# Set the workspace config path for this batch run.
CONFIG=/abs/path/to/workspace/config.yaml

# Validate schema and solver availability before queue submission.
uv run dense validate-config --probe-solver -c "$CONFIG"

# Inspect resolved config to verify sink/source and runtime limits.
uv run dense inspect config --probe-solver -c "$CONFIG"
```

### Batch execution pattern
This section provides the recommended command sequence for long-running jobs.

```bash
# Build Stage-A pools explicitly so failures happen early.
uv run dense stage-a build-pool --fresh -c "$CONFIG"

# Run DenseGen without plotting during the batch window.
uv run dense run --no-plot -c "$CONFIG"

# Inspect run diagnostics and library summaries.
uv run dense inspect run --events --library -c "$CONFIG"

# Render plots and notebook after core generation succeeds.
uv run dense plot -c "$CONFIG"
uv run dense notebook generate -c "$CONFIG"
```

### Resume pattern
This section documents safe continuation behavior for iterative campaigns.

```bash
# Resume from existing run state.
uv run dense run --resume --no-plot -c "$CONFIG"

# Resume and extend quotas for additional sampling.
uv run dense run --resume --extend-quota 8 --no-plot -c "$CONFIG"
```

### Notify wiring
This section points to the canonical event-boundary docs instead of duplicating event semantics.

Notify must consume USR `.events.log` rather than DenseGen diagnostics. Use **[observability and events](../concepts/observability_and_events.md)** plus the **[DenseGen to USR to Notify tutorial](../tutorials/demo_usr_notify.md)** for wiring and validation.
