## DenseGen TFBS baseline tutorial

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-28


This tutorial runs the smallest DenseGen workflow with binding-site inputs and local parquet outputs.

### Runbook command

Use the workspace runbook for the command sequence: [demo_tfbs_baseline/runbook.md](../../workspaces/demo_tfbs_baseline/runbook.md).

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/demo_tfbs_baseline
# Execute the packaged workspace runbook sequence.
./runbook.sh
```

### Prerequisites

```bash
# Install locked Python dependencies for reproducible execution.
uv sync --locked
# Confirm the DenseGen CLI is installed and discoverable.
uv run dense --help
# Validate config schema and probe solver availability.
uv run dense validate-config --probe-solver -c src/dnadesign/densegen/workspaces/demo_tfbs_baseline/config.yaml
```

### Key config sections

```yaml
# src/dnadesign/densegen/workspaces/demo_tfbs_baseline/config.yaml
densegen:
  output:
    targets: [parquet]                   # Write local records table only.
  generation:
    sequence_length: 100                 # Final sequence length for both plans.
    sampling:
      pool_strategy: iterative_subsample  # Stage-A/Stage-B pool sampling strategy.
      iterative_max_libraries: 200        # Upper bound on Stage-B library resamples.
    plan:
      - name: baseline                    # Plan identifier.
        sequences: 50                     # Unconstrained plan quota.
      - name: baseline_sigma70            # Plan identifier.
        sequences: 50                     # Sigma70-constrained plan quota.
        fixed_elements:
          promoter_constraints:
            - name: sigma70_consensus     # Constraint identifier.
              spacer_length: [16, 18]     # Enforce spacing between -35 and -10 motifs.
  solver:
    backend: CBC                          # Dense-arrays backend.
    strategy: iterate                     # Iterative solve strategy.
  runtime:
    round_robin: true                     # Keep sampling rounds until quotas are met.
    max_accepted_per_library: 10          # Stage-B attempts before resampling.
    max_failed_solutions: 0               # Disable failed-solution accumulation.
```

### Step-by-step commands

Set workspace-local paths first so each command resolves against the same config.

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/demo_tfbs_baseline
# Pin config path for repeated CLI calls.
CONFIG="$PWD/config.yaml"
```

Run generation first, then inspect progress before rendering analysis artifacts.

```bash
# Validate config schema and probe solver availability.
uv run dense validate-config --probe-solver -c "$CONFIG"
# Start a fresh run from a clean output state.
uv run dense run --fresh --no-plot -c "$CONFIG"
# Inspect run diagnostics and per-plan library progress.
uv run dense inspect run --events --library -c "$CONFIG"
```

Render plots and notebook outputs only after the run passes validation and inspection.

```bash
# Render DenseGen analysis artifacts from current run outputs.
# `dense plot` is the analysis entry point; static plots always render.
# Set plots.video.enabled: true in config to also emit a sampled Stage-B showcase video
# at outputs/plots/stage_b/all_plans/showcase.mp4.
uv run dense plot -c "$CONFIG"
# Generate the run-overview marimo notebook artifact.
uv run dense notebook generate -c "$CONFIG"
# Run notebook validation before opening or sharing it.
uv run marimo check "$PWD/outputs/notebooks/densegen_run_overview.py"
```

```bash
# Optional analysis shortcut: render only the Stage-B showcase video artifact.
# uv run dense plot --only dense_array_video_showcase -c "$CONFIG"
```

### If outputs already exist (analysis-only)

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/demo_tfbs_baseline
# Rebuild plots/notebook from existing run artifacts without regenerating sequences.
./runbook.sh --analysis-only
# Open the generated notebook in marimo app mode.
uv run dense notebook run -c "$PWD/config.yaml"
```

### Expected outputs

- `outputs/tables/records.parquet`
- `outputs/meta/events.jsonl`
- `outputs/plots/`
- `outputs/notebooks/densegen_run_overview.py`

### Related docs

- [Pipeline lifecycle](../concepts/pipeline-lifecycle.md)
- [Outputs reference](../reference/outputs.md)
- [Workspace catalog](../../workspaces/catalog.md)
