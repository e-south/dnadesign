## DenseGen quick checklist

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


This concept page is a compact contract checklist for correct DenseGen execution and artifact interpretation. Read it when you need quick command recall and clear output-path expectations.

For full walkthroughs, use the **[TFBS baseline tutorial](../tutorials/demo_tfbs_baseline.md)** or **[sampling baseline tutorial](../tutorials/demo_sampling_baseline.md)**.

### Core command flow
This section provides the smallest reliable command set for a workspace lifecycle.

```bash
# Resolve repo root, then pin workspace root and derive config path once.
REPO_ROOT="$(git rev-parse --show-toplevel)"
WORKSPACE_ROOT="$REPO_ROOT/src/dnadesign/densegen/workspaces"

# Initialize a workspace from a packaged template.
uv run dense workspace init --id <workspace_id> --root "$WORKSPACE_ROOT" --from-workspace <template_id> --copy-inputs --output-mode <local|usr|both>

# Use this config path for subsequent commands.
CONFIG="$WORKSPACE_ROOT/<workspace_id>/config.yaml"

# Validate config and probe solver availability.
uv run dense validate-config --probe-solver -c "$CONFIG"

# Build Stage-A pools when inputs require mining.
uv run dense stage-a build-pool --fresh -c "$CONFIG"

# Run solve-to-quota.
uv run dense run -c "$CONFIG"

# Inspect runtime events and library summaries.
uv run dense inspect run --events --library -c "$CONFIG"

# Render plots and generate notebook.
uv run dense plot -c "$CONFIG"
uv run dense notebook generate -c "$CONFIG"
```

### Artifact paths
This section lists the file paths you should memorize for fast diagnostics.

- DenseGen diagnostics: `outputs/meta/events.jsonl`
- Stage-A pools: `outputs/pools/` and `outputs/pools/pool_manifest.json`
- Stage-B libraries: `outputs/libraries/`
- Parquet records: `outputs/tables/records.parquet`
- USR records: `outputs/usr_datasets/<dataset>/records.parquet`
- USR event stream: `outputs/usr_datasets/<dataset>/.events.log`

### Output mode rules
This section explains which sinks are produced for each output mode.

- `local` writes parquet artifacts only.
- `usr` writes USR dataset artifacts only.
- `both` writes both sinks and requires explicit `plots.source` selection for analysis surfaces.

### Resume rules
This section states the safe-state rules for iterative runs.

- Resume with unchanged config: `dense run --resume`.
- Resume with runtime quota growth: `dense run --resume --extend-quota <n>`.
- Restart from clean output state: `dense run --fresh`.

### Event stream rules
This section points to the full explanation instead of duplicating it.

DenseGen diagnostics and USR mutation events are distinct streams with distinct consumers. Read **[observability and events](observability_and_events.md)** before wiring Notify.
