## DenseGen sampling baseline tutorial

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-28


Use this tutorial to run the sampling baseline with PWM inputs. It covers Stage-A mining, Stage-B generation, and dual output sinks (`parquet` and `usr`).

### Runbook command

Use the workspace runbook sequence from [demo_sampling_baseline/runbook.md](../../workspaces/demo_sampling_baseline/runbook.md). This command runs a clean pass through validation, generation, inspection, and analysis rendering.

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/demo_sampling_baseline
# Run the packaged flow in explicit fresh mode.
./runbook.sh --mode fresh
```

Use `--mode resume` to continue generation, or `--mode analysis` when you only need plots/notebook refresh.

### Prerequisites

Run these once to install dependencies and verify required CLI tools are available.

```bash
# Install locked Python dependencies for reproducible execution.
uv sync --locked
# Install pixi-managed tooling required by this workflow.
pixi install
# Verify FIMO is available before PWM-backed sampling/validation.
pixi run fimo --version
# Validate config schema and probe solver availability.
pixi run dense validate-config --probe-solver -c src/dnadesign/densegen/workspaces/demo_sampling_baseline/config.yaml
```

### Key config sections

```yaml
# src/dnadesign/densegen/workspaces/demo_sampling_baseline/config.yaml
densegen:                                   # DenseGen runtime settings root.
  output:                                   # Output sink settings.
    targets: [parquet, usr]                 # Output sinks; options include parquet and usr.
  generation:                               # Sequence-generation controls.
    sequence_length: 100                    # Final sequence length in base pairs.
    sampling:                               # Stage-A and Stage-B sampling controls.
      pool_strategy: subsample              # Pool strategy; common values are subsample or iterative_subsample.
      library_size: 10                      # Candidate library size sampled per plan.
    plan:                                   # List of generation plans.
      - name: ethanol                       # Plan name shown in outputs and summaries.
        sequences: 6                        # Target sequence count for this plan.
      - name: ciprofloxacin                 # Plan name shown in outputs and summaries.
        sequences: 6                        # Target sequence count for this plan.
  runtime:                                  # Runtime stop and retry settings.
    max_failed_solutions_per_target: 2.0    # Failed-solve budget scaled by target count.
```

```yaml
# Stage-A sampling profile (same config file)
inputs:                                     # Input definitions used by Stage-A and filters.
  - name: lexA_pwm                          # Input name referenced by plans and filters.
    sampling:                               # Sampling controls for this input.
      n_sites: 100                          # Number of retained Stage-A sites for this input.
      mining:                               # Candidate-mining controls before retention.
        batch_size: 2000                    # Candidates evaluated per mining batch.
        budget:                             # Budget policy for Stage-A mining.
          mode: fixed_candidates            # Budget mode; fixed_candidates uses a hard candidate cap.
          candidates: 150000                # Total candidate budget for this input.
```

### Step-by-step commands

Start by pinning the config path used across run commands. This workspace writes local tables to `outputs/tables/` and USR outputs to `outputs/usr_datasets/`.
`dense run` auto-seeds `outputs/usr_datasets/registry.yaml` when it is missing, so no manual registry copy step is required.

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/demo_sampling_baseline
# Pin config path for repeated CLI calls.
CONFIG="$PWD/config.yaml"
```

Use this block to validate dependencies, run generation, and inspect progress before plotting.

```bash
# Verify FIMO is available before PWM-backed sampling/validation.
pixi run fimo --version
# Validate config schema and probe solver availability.
pixi run dense validate-config --probe-solver -c "$CONFIG"
# Start a fresh run from a clean output state.
pixi run dense run --fresh --no-plot -c "$CONFIG"
# Inspect run diagnostics and per-plan library progress.
pixi run dense inspect run --events --library -c "$CONFIG"
```

Use this block to render plots and generate the notebook from current outputs.

```bash
# Render DenseGen analysis artifacts from current run outputs.
# `dense plot` is the analysis entry point; static plots always render.
# Set plots.video.enabled: true in config to also emit a sampled Stage-B showcase video
# at outputs/plots/stage_b/all_plans/showcase.mp4.
pixi run dense plot -c "$CONFIG"
# Generate the run-overview marimo notebook artifact.
pixi run dense notebook generate -c "$CONFIG"
# Run notebook validation before opening or sharing it.
uv run marimo check "$PWD/outputs/notebooks/densegen_run_overview.py"
```

```bash
# Optional analysis shortcut: render only the Stage-B showcase video artifact.
# pixi run dense plot --only dense_array_video_showcase -c "$CONFIG"
```

### If outputs already exist (analysis mode)

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/demo_sampling_baseline
# Rebuild plots/notebook from existing run artifacts without regenerating sequences.
./runbook.sh --mode analysis
# Open the generated notebook in marimo app mode.
pixi run dense notebook run -c "$PWD/config.yaml"
```

### Optional artifact refresh from Cruncher

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/demo_sampling_baseline
# Export Cruncher motif artifacts into the DenseGen workspace.
uv run cruncher catalog export-densegen --set 1 --source demo_merged_meme_oops_multitf --densegen-workspace "$PWD" -c "$(git rev-parse --show-toplevel)/src/dnadesign/cruncher/workspaces/demo_multitf/configs/config.yaml"
```

### Expected outputs

- `outputs/pools/pool_manifest.json`
- `outputs/tables/records.parquet`
- `outputs/usr_datasets/densegen/demo_sampling_baseline/.events.log`
- `outputs/plots/`
- `outputs/notebooks/densegen_run_overview.py`

### Related docs

- [Sampling concept](../concepts/sampling.md)
- [Outputs reference](../reference/outputs.md)
- [Workspaces directory](../../workspaces/README.md)
