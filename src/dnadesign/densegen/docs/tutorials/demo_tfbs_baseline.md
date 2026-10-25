## DenseGen TFBS baseline tutorial

**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-25


Use this tutorial to run the smallest DenseGen workspace end to end. You validate the config, generate sequences from TFBS inputs, and render analysis artifacts from local outputs.

### Prerequisites

From the repository root, install dependencies and enter the packaged workspace
once. The remaining commands run from that directory.

```bash
# Install the locked tool dependencies.
uv sync --locked
# Enter the packaged workspace.
cd src/dnadesign/densegen/workspaces/demo_tfbs_baseline
# Reuse this config for the individual commands below.
CONFIG="$PWD/config.yaml"
# Validate the input and check solver availability.
uv run dense validate-config --probe-solver -c "$CONFIG"
```

### Runbook command

The [workspace runbook](../../workspaces/demo_tfbs_baseline/runbook.md) validates
inputs, generates sequences, and creates plots and a notebook:

```bash
# Run validation, generation and analysis.
./runbook.sh --mode fresh
```

Choose this runbook or the individual commands below. Use `--mode resume` to
continue generation, or `--mode analysis` to update plots and the notebook from
existing outputs.

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

Use the `CONFIG` path set above.

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
# The Stage-B showcase video is optional. Enable plots.video and either add
# dense_array_showcase_video to plots.default or render it explicitly with --only.
uv run dense plot -c "$CONFIG"
# Generate the run-overview marimo notebook artifact.
uv run dense notebook generate -c "$CONFIG"
# Run notebook validation before opening or sharing it.
uv run marimo check "$PWD/outputs/notebooks/densegen_run_overview.py"
```

```bash
# Optional analysis shortcut: render only the Stage-B showcase video artifact.
# uv run dense plot --only dense_array_showcase_video -c "$CONFIG"
```

### If outputs already exist (analysis mode)

```bash
# Rebuild plots/notebook from existing run artifacts without regenerating sequences.
./runbook.sh --mode analysis
# Open the generated notebook in marimo app mode.
uv run dense notebook run -c "$PWD/config.yaml"
```

### Expected outputs

The configuration requests 50 sequences for each of two plans, all 100 bases
long. `baseline_sigma70` also requires the specified sigma70 promoter elements.
Use `dense inspect run` to check the achieved counts before interpreting plots.

- `outputs/tables/records.parquet`
- `outputs/meta/events.jsonl`
- `outputs/plots/`
- `outputs/notebooks/densegen_run_overview.py`

### Related docs

- [Pipeline lifecycle](../concepts/runtime/pipeline-lifecycle.md)
- [Outputs reference](../reference/outputs.md)
- [Workspaces directory](../../workspaces/README.md)
