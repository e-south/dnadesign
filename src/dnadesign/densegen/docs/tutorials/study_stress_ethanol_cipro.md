## Stress ethanol and ciprofloxacin study tutorial

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


This tutorial runs the largest packaged DenseGen campaign with expanded stress-condition plans and USR-ready outputs.

### Runbook command

Use the workspace runbook for the command sequence: [study_stress_ethanol_cipro/runbook.md](../../workspaces/study_stress_ethanol_cipro/runbook.md).

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro
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
pixi run dense validate-config --probe-solver -c src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/config.yaml
```

### Key config sections

```yaml
# src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/config.yaml
densegen:
  output:
    targets: [parquet, usr]      # Keep analysis tables and USR events for Notify.
  generation:
    sequence_length: 60                   # Final designed sequence length in base pairs.
    expansion:
      max_plans: 64                       # Upper bound on expanded plan variants.
    sampling:
      library_size: 10                    # Sampled library breadth per plan.
    plan:
      - name: ethanol                     # Identifier for this config entry.
        sequences: 60                     # Requested sequence quota for this plan.
      - name: ciprofloxacin               # Identifier for this config entry.
        sequences: 60                     # Requested sequence quota for this plan.
      - name: ethanol_ciprofloxacin       # Identifier for this config entry.
        sequences: 80                     # Requested sequence quota for this plan.
  runtime:
    max_failed_solutions_per_target: 2.0  # Failed-solve tolerance scaled by target count.
```

```yaml
# Stage-A campaign mining profile (same config file)
inputs:
  - name: lexA_pwm                # Identifier for this config entry.
    sampling:
      n_sites: 250                # Number of retained sampled sites.
      mining:
        batch_size: 5000          # Candidate count evaluated per mining batch.
        budget:
          mode: fixed_candidates  # Budget policy for mining candidates.
          candidates: 1000000    # PWM mining effort per regulator.
  - name: background              # Identifier for this config entry.
    sampling:
      n_sites: 500                # Number of retained sampled sites.
      mining:
        batch_size: 20000         # Candidate count evaluated per mining batch.
        budget:
          mode: fixed_candidates  # Budget policy for mining candidates.
          candidates: 5000000    # Background mining budget under exclusion filters.
```

### Step-by-step commands

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro
# Pin config path for repeated CLI calls.
CONFIG="$PWD/config.yaml"
# Pin workspace-local USR registry destination.
USR_REGISTRY="$PWD/outputs/usr_datasets/registry.yaml"
# Resolve repo-level baseline USR registry path.
ROOT_REGISTRY="$(git rev-parse --show-toplevel)/src/dnadesign/usr/datasets/registry.yaml"

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

### Optional artifact refresh from Cruncher

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro
# Export Cruncher motif artifacts into the DenseGen workspace.
uv run cruncher catalog export-densegen --set 1 --source demo_merged_meme_oops_multitf --densegen-workspace "$PWD" -c "$(git rev-parse --show-toplevel)/src/dnadesign/cruncher/workspaces/demo_multitf/configs/config.yaml"
```

### Expected outputs

- `outputs/tables/records.parquet`
- `outputs/usr_datasets/study_stress_ethanol_cipro/.events.log`
- `outputs/plots/`
- `outputs/notebooks/densegen_run_overview.py`

### Related docs

- [DenseGen to USR to Notify tutorial](demo_usr_notify.md)
- [Outputs reference](../reference/outputs.md)
- [Workspace catalog](../../workspaces/catalog.md)
