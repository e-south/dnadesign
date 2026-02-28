## DenseGen HPC runbook

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-28


This how-to guide provides scheduler-safe DenseGen execution patterns for long-running environments. Read it when you need predictable preflight checks, resume behavior, and clear separation between generation and analysis phases.

### When to use this runbook
This section clarifies scope so operators can choose the right guide.

Use this runbook for generic scheduler workflows; use **[DenseGen on BU SCC](bu-scc.md)** for BU-specific flags and policy details.

For expected artifact paths while debugging batch runs, use **[DenseGen outputs reference](../reference/outputs.md)**.

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

### Core generation flow (run shell or interactive session)
This section covers generation commands that work the same way in a direct shell, an interactive remote session, or inside a batch script.

```bash
# Build Stage-A pools explicitly so failures happen early.
uv run dense stage-a build-pool --fresh -c "$CONFIG"
# Start a clean generation branch.
uv run dense run --fresh --no-plot -c "$CONFIG"
# Resume generation without wiping outputs.
uv run dense run --resume --no-plot -c "$CONFIG"
# Increase total quota for this run without editing config.yaml.
uv run dense run --resume --extend-quota 50000 --no-plot -c "$CONFIG"
# Inspect run diagnostics and plan-level progress.
uv run dense inspect run --events --library -c "$CONFIG"
```

### Scheduler submission flow (batch wrapper)
This section wraps the same generation flow in scheduler submission commands. For BU SCC-specific flags and policy, use **[DenseGen on BU SCC](bu-scc.md)** and the repo-level SCC docs.

```bash
# Check current scheduler pressure before proposing new submissions.
qstat -u "$USER"
# Summarize running, queued, and Eqw jobs for submit gating.
qstat -u "$USER" | awk '$1 ~ /^[0-9]+$/ { running += ($5 ~ /r/); queued += ($5 ~ /q/); eqw += ($5 ~ /Eqw/) } END { printf "running_jobs=%d queued_jobs=%d eqw_jobs=%d\n", running, queued, eqw }'
# Submit generation-only run against a workspace config.
qsub -P <project> -v DENSEGEN_CONFIG="$CONFIG",DENSEGEN_RUN_ARGS='--resume --no-plot' docs/bu-scc/jobs/densegen-cpu.qsub
# Submit a bounded extension pass for iterative quota growth.
qsub -P <project> -v DENSEGEN_CONFIG="$CONFIG",DENSEGEN_RUN_ARGS='--resume --extend-quota 50000 --no-plot' docs/bu-scc/jobs/densegen-cpu.qsub
```

Queue-fair policy:
- If `running_jobs > 3`, avoid burst submits and prefer arrays or `-hold_jid` chains.
- Respect scheduler order and do not use queue-bypass behavior.

Single-workspace concurrency contract:
- DenseGen keeps one active writer per workspace/run root using `outputs/meta/run.lock`.
- Concurrent submits against the same workspace are expected to fail fast with a lock-held error.
- For repeated contributions on one workspace, prefer `-hold_jid` chains.
- For true parallel exploration (for example Stage-A mining/diversity variants), branch to separate workspaces/run roots and merge approved USR datasets later with `uv run usr maintenance merge`.

### Post-run analysis flow
This section runs analysis after generation is complete (or paused), without modifying sequence outputs.

```bash
# Render DenseGen analysis artifacts from current outputs.
uv run dense plot -c "$CONFIG"
# Optional analysis shortcut: render only the Stage-B showcase video artifact.
# uv run dense plot --only dense_array_video_showcase -c "$CONFIG"
# Generate the run-overview marimo notebook artifact.
uv run dense notebook generate -c "$CONFIG"
# Validate the generated notebook before sharing.
uv run marimo check "$(dirname "$CONFIG")/outputs/notebooks/densegen_run_overview.py"
```

### Config-change guardrails for resume safety
This section defines which config edits are allowed in-place when resuming.

- In-place resume accepts quota-only config changes (for example `generation.plan[].sequences` increases).
- Non-quota changes are blocked in-place with `Config changed beyond plan quotas.`.
- For Stage-A sampling/diversity edits, sequence-length edits, fixed-element edits, or solver edits, use a new run root or rerun with `--fresh`.

Durability knobs for interruption tolerance:
- `densegen.runtime.checkpoint_every` controls checkpoint and sink flush cadence.
- `densegen.output.parquet.chunk_size` and `densegen.output.usr.chunk_size` control buffered write size per flush.
- Lower values reduce potential in-memory loss on hard interruption, with higher I/O overhead.

### Notify event wiring
This section points to the event-boundary docs instead of duplicating event semantics.

Notify must consume USR `.events.log` rather than DenseGen diagnostics. Use **[observability and events](../concepts/observability_and_events.md)** plus the **[DenseGen to USR to Notify tutorial](../tutorials/demo_usr_notify.md)** for wiring and validation.
