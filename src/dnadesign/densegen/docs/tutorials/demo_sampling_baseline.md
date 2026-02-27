## DenseGen sampling baseline tutorial

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


This tutorial runs the sampling baseline with PWM artifacts, Stage-A mining, Stage-B generation, and dual-sink outputs.

### Runbook command

Use the workspace runbook for the command sequence: [demo_sampling_baseline/runbook.md](../../workspaces/demo_sampling_baseline/runbook.md).

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/demo_sampling_baseline
# Execute the packaged workspace runbook sequence.
./runbook.sh
```

### Prerequisites

```bash
# Install locked Python dependencies for reproducible execution.
uv sync --locked
# Install pixi-managed tooling required by this workflow.
pixi install
# Verify FIMO is available before PWM-backed sampling/validation.
pixi run fimo --version
# Validate config schema and probe solver availability.
pixi run dense validate-config --probe-solver -c src/dnadesign/densegen/workspaces/demo_sampling_baseline/config.yaml
# Create the target directory if it does not already exist.
mkdir -p src/dnadesign/densegen/workspaces/demo_sampling_baseline/outputs/usr_datasets
# Copy baseline artifacts into the workspace-local location.
cp src/dnadesign/usr/datasets/registry.yaml src/dnadesign/densegen/workspaces/demo_sampling_baseline/outputs/usr_datasets/registry.yaml
```

### Key config sections

```yaml
# src/dnadesign/densegen/workspaces/demo_sampling_baseline/config.yaml
densegen:
  output:
    targets: [parquet, usr]              # Keep local tables and USR event dataset in one run.
  generation:
    sequence_length: 100                  # Final sequence length for both plans.
    sampling:
      pool_strategy: subsample            # Stage-A/Stage-B pool sampling strategy.
      library_size: 10                    # Stage-B library breadth per plan.
    plan:
      - name: ethanol                     # Plan identifier.
        sequences: 6                      # Requested sequence quota for this plan.
      - name: ciprofloxacin               # Plan identifier.
        sequences: 6                      # Requested sequence quota for this plan.
  runtime:
    max_failed_solutions_per_target: 2.0  # Failed-solve tolerance scaled by target count.
```

```yaml
# Stage-A sampling profile (same config file)
inputs:
  - name: lexA_pwm                        # Input identifier.
    sampling:
      n_sites: 100                         # Retained Stage-A sites for this regulator.
      mining:
        batch_size: 2000                   # Candidate count evaluated per mining batch.
        budget:
          mode: fixed_candidates           # Budget policy for mining candidates.
          candidates: 150000               # Mining effort cap per regulator.
```

### Step-by-step commands

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/demo_sampling_baseline
# Pin config path for repeated CLI calls.
CONFIG="$PWD/config.yaml"
# Pin workspace-local USR registry destination.
USR_REGISTRY="$PWD/outputs/usr_datasets/registry.yaml"
# Resolve repo-level baseline USR registry path.
ROOT_REGISTRY="$(git rev-parse --show-toplevel)/src/dnadesign/usr/datasets/registry.yaml"

# Seed a workspace-local USR registry when one is not present.
if [ ! -f "$USR_REGISTRY" ]; then
  # Create the target directory if it does not already exist.
  mkdir -p "$(dirname "$USR_REGISTRY")"
  # Copy baseline artifacts into the workspace-local location.
  cp "$ROOT_REGISTRY" "$USR_REGISTRY"
fi

# Verify FIMO is available before PWM-backed sampling/validation.
pixi run fimo --version
# Validate config schema and probe solver availability.
pixi run dense validate-config --probe-solver -c "$CONFIG"
# Start a fresh run from a clean output state.
pixi run dense run --fresh --no-plot -c "$CONFIG"
# Inspect run diagnostics and per-plan library progress.
pixi run dense inspect run --events --library -c "$CONFIG"
# Render DenseGen plots from current run artifacts.
pixi run dense plot -c "$CONFIG"
# Generate the run-overview marimo notebook artifact.
pixi run dense notebook generate -c "$CONFIG"
# Run notebook validation before opening or sharing it.
uv run marimo check "$PWD/outputs/notebooks/densegen_run_overview.py"
```

### If outputs already exist (analysis-only)

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/demo_sampling_baseline
# Rebuild plots/notebook from existing run artifacts without regenerating sequences.
./runbook.sh --analysis-only
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
- `outputs/usr_datasets/demo_sampling_baseline/.events.log`
- `outputs/plots/`
- `outputs/notebooks/densegen_run_overview.py`

### Related docs

- [Sampling concept](../concepts/sampling.md)
- [Outputs reference](../reference/outputs.md)
- [Workspace catalog](../../workspaces/catalog.md)
