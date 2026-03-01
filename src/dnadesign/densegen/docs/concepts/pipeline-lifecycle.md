## DenseGen pipeline lifecycle

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-28
This concept document explains the DenseGen lifecycle as an operator sequence from workspace initialization through run artifacts and reset. Read it when you need a stable mental model for debugging stage failures, resume behavior, and output placement.

### Lifecycle summary

1. Initialize workspace from a packaged template.
2. Validate config and solver availability.
3. Build Stage-A pools when inputs require mining.
4. Run DenseGen solve-to-quota.
5. Inspect run artifacts and diagnostics.
6. Render plots and notebook outputs.
7. Resume or reset intentionally.

If you want copy/paste commands for this flow, use **[DenseGen quick checklist](quick-checklist.md)**.

### Pipeline stages

| Stage | Purpose | Typical failure surface |
| --- | --- | --- |
| Stage-A | Build/load candidate pools from configured inputs. | Missing external tooling, empty pools, overly strict selection filters. |
| Stage-B | Build plan-scoped libraries from Stage-A pools. | Over-constrained plans, insufficient diversity, repeated candidate collisions. |
| Solve | Place motifs/fixed elements to satisfy quotas. | Infeasible constraints, solver backend errors, strict runtime caps. |
| Post-run | Materialize records, metadata, plots, and notebooks. | Sink/source mismatch, missing artifact source paths, notebook source confusion. |

### Key artifact paths

- `outputs/meta/events.jsonl` for DenseGen runtime diagnostics.
- `outputs/pools/pool_manifest.json` for Stage-A pool inventory.
- `outputs/libraries/` for Stage-B library membership and summaries.
- `outputs/tables/records.parquet` when parquet sink is enabled.
- `outputs/usr_datasets/<dataset>/records.parquet` when USR sink is enabled.

### Output modes and analysis source selection

- `local`: writes local parquet outputs only.
- `usr`: writes USR dataset outputs only.
- `both`: writes both sinks and requires explicit source selection for analysis commands.

When both sinks are enabled, plots and notebooks resolve records from `plots.source`.

### Resume and reset behavior

- `dense run --resume` continues from existing run state.
- `dense run --resume --extend-quota <n>` grows quotas without editing config.
- `dense run --fresh` clears outputs and restarts from clean state.
- `dense campaign-reset` clears outputs while preserving config and inputs.

### Strand handling by stage

- Stage-A scoring and filtering semantics depend on the configured input/scoring backend.
- Solver strand model follows config-level generation constraints.
- Final sequence acceptance is controlled by configured sequence constraints.

For exact event-stream boundaries and consumer contracts, read **[observability and events](observability_and_events.md)**. For scheduler execution patterns, use **[DenseGen HPC runbook](../howto/hpc.md)**.
