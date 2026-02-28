## DenseGen TFBS baseline tutorial

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-28


Use this tutorial to run the smallest DenseGen workspace end to end. You validate the config, generate sequences from TFBS inputs, and render analysis artifacts from local outputs.

### Runbook command

Use the workspace runbook sequence from [demo_tfbs_baseline/runbook.md](../../workspaces/demo_tfbs_baseline/runbook.md). This command runs a clean pass from validation through notebook generation.

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/demo_tfbs_baseline
# Run the packaged flow in explicit fresh mode.
./runbook.sh --mode fresh
```

Use `--mode resume` to continue generation, or `--mode analysis` when you only need plots/notebook refresh.

### Prerequisites

Run these once to install dependencies and confirm the CLI and solver path are available.

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
densegen:                                   # DenseGen runtime settings root.
  output:                                   # Output sink settings.
    targets: [parquet]                      # Output sinks; options include parquet and usr.
  generation:                               # Sequence-generation controls.
    sequence_length: 100                    # Final sequence length in base pairs.
    sampling:                               # Stage-A and Stage-B sampling controls.
      pool_strategy: iterative_subsample    # Pool strategy; common values are iterative_subsample or subsample.
      iterative_max_libraries: 200          # Max Stage-B library retries when quotas are not met.
    plan:                                   # List of generation plans.
      - name: baseline                      # Plan name shown in outputs and summaries.
        sequences: 50                       # Target sequence count for this plan.
      - name: baseline_sigma70              # Plan name for sigma-constrained generation.
        sequences: 50                       # Target sequence count for this plan.
        fixed_elements:                     # Fixed motif constraints used by this plan.
          promoter_constraints:             # Promoter-core constraints to enforce.
            - name: sigma70_consensus       # Constraint label used in logs and metadata.
              spacer_length: [16, 18]       # Allowed spacer range between -35 and -10 motifs.
  solver:                                   # Dense-array solver settings.
    backend: CBC                            # Solver backend; common backends include CBC and GUROBI.
    strategy: iterate                       # Solver strategy; iterate performs repeated bounded passes.
  runtime:                                  # Runtime stop and retry settings.
    round_robin: true                       # True cycles through plans until quotas are met.
    max_accepted_per_library: 10            # Accepted rows per library draw before resampling.
    max_failed_solutions: 0                 # Failed-solve cap; 0 disables accumulation.
```

### Step-by-step commands

Set the config path once so every command in this section runs against the same workspace.

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/demo_tfbs_baseline
# Pin config path for repeated CLI calls.
CONFIG="$PWD/config.yaml"
```

Run generation first, then inspect progress. Plot and notebook steps come after generation succeeds.

```bash
# Validate config schema and probe solver availability.
uv run dense validate-config --probe-solver -c "$CONFIG"
# Start a fresh run from a clean output state.
uv run dense run --fresh --no-plot -c "$CONFIG"
# Inspect run diagnostics and per-plan library progress.
uv run dense inspect run --events --library -c "$CONFIG"
```

Use this block to render plots and generate the notebook from the current outputs.

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

### If outputs already exist (analysis mode)

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/demo_tfbs_baseline
# Rebuild plots/notebook from existing run artifacts without regenerating sequences.
./runbook.sh --mode analysis
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
- [Workspaces directory](../../workspaces/README.md)
