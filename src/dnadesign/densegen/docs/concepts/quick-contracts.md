## DenseGen quick contracts

This concept page is a compact contract checklist for correct DenseGen execution and artifact interpretation. Read it when you need fast command recall and canonical output-path expectations.

### Core command sequence
This section provides the smallest reliable command set for a workspace lifecycle.

```bash
# Initialize a workspace from a packaged template.
uv run dense workspace init --id <workspace_id> --from-workspace <template_id> --copy-inputs --output-mode <local|usr|both>

# Validate config and probe solver availability.
uv run dense validate-config --probe-solver -c <workspace>/config.yaml

# Build Stage-A pools when inputs require mining.
uv run dense stage-a build-pool --fresh -c <workspace>/config.yaml

# Run solve-to-quota.
uv run dense run -c <workspace>/config.yaml

# Inspect runtime events and library summaries.
uv run dense inspect run --events --library -c <workspace>/config.yaml

# Render plots and generate notebook.
uv run dense plot -c <workspace>/config.yaml
uv run dense notebook generate -c <workspace>/config.yaml
```

### Canonical artifact paths
This section lists the file paths you should memorize for fast diagnostics.

- DenseGen diagnostics: `outputs/meta/events.jsonl`
- Stage-A pools: `outputs/pools/` and `outputs/pools/pool_manifest.json`
- Stage-B libraries: `outputs/libraries/`
- Parquet records: `outputs/tables/records.parquet`
- USR records: `outputs/usr_datasets/<dataset>/records.parquet`
- USR event stream: `outputs/usr_datasets/<dataset>/.events.log`

### Output mode contract
This section explains which sinks are produced for each output mode.

- `local` writes parquet artifacts only.
- `usr` writes USR dataset artifacts only.
- `both` writes both sinks and requires explicit `plots.source` selection for analysis surfaces.

### Resume contract
This section states the safe-state rules for iterative runs.

- Resume with unchanged config: `dense run --resume`.
- Resume with runtime quota growth: `dense run --resume --extend-quota <n>`.
- Restart from clean output state: `dense run --fresh`.

### Event stream contract
This section points to the canonical explanation instead of duplicating it.

DenseGen diagnostics and USR mutation events are distinct streams with distinct consumers. Read **[observability and events](observability_and_events.md)** before wiring Notify.
