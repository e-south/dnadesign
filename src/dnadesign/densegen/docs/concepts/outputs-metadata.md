## Outputs and metadata concepts

This concept page explains how DenseGen artifacts are organized by runtime stage and how to interpret them together. Read it when you need to debug run outcomes, join tables correctly, or choose the right file for analysis.

### Output layout mental model
This section gives a stable directory model for understanding where each stage writes.

- `outputs/pools/` stores Stage-A retained pools and pool manifests.
- `outputs/libraries/` stores Stage-B library build artifacts.
- `outputs/tables/` stores run tables including records, attempts, and diagnostics.
- `outputs/meta/` stores manifests, effective config, and runtime events.
- `outputs/plots/` and `outputs/notebooks/` store analysis artifacts.

### Stage-to-artifact mapping
This section maps each stage boundary to the artifacts that are authoritative for that stage.

- Stage-A authority: `outputs/pools/pool_manifest.json` and per-input pool parquet files.
- Stage-B authority: `outputs/libraries/library_manifest.json` plus library membership tables.
- Solve authority: `outputs/tables/attempts.parquet`, `outputs/tables/solutions.parquet`, and `outputs/tables/records.parquet`.
- Runtime diagnostics authority: `outputs/meta/events.jsonl` and `outputs/tables/run_metrics.parquet`.

### Canonical records surface
This section explains where final records live for each output mode.

- Parquet sink: `outputs/tables/records.parquet`.
- USR sink: `outputs/usr_datasets/<dataset>/records.parquet`.
- Dual sink mode: both files exist; analysis commands resolve one source via `plots.source`.

### Join-key expectations
This section summarizes how to join dense output tables without ambiguity.

- Use stable record identifiers from `records.parquet` for run-level joins.
- Use placement-level keys from composition tables for motif placement analysis.
- Prefer manifest metadata over inferred assumptions when correlating stage outputs.

### Event-stream boundary
This section links to the canonical event boundary explanation to avoid duplication.

Read **[observability and events](observability_and_events.md)** for DenseGen diagnostics versus USR mutation events and Notify consumption rules.

### Contract-grade details
This section points to exact schemas and field-level definitions.

- Use **[outputs reference](../reference/outputs.md)** for artifact contracts.
- Use **[config reference](../reference/config.md)** for sink and plotting source keys.
