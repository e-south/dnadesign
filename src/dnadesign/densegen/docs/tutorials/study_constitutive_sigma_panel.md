## Constitutive sigma70 panel tutorial

This tutorial runs the constitutive sigma70 panel workspace with fixed-element expansion and strict LacI/AraC background filtering.

### Runbook command

Use the workspace runbook for the command sequence: [study_constitutive_sigma_panel/runbook.md](../../workspaces/study_constitutive_sigma_panel/runbook.md).

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/study_constitutive_sigma_panel
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
pixi run dense validate-config --probe-solver -c src/dnadesign/densegen/workspaces/study_constitutive_sigma_panel/config.yaml
```

### Key config sections

```yaml
# src/dnadesign/densegen/workspaces/study_constitutive_sigma_panel/config.yaml
densegen:
  generation:
    sequence_length: 60          # Final sequence length for the panel.
    expansion:
      max_plans: 64              # Guardrail for expanded plan count.
    plan:
      - name: sigma70_panel          # Identifier for this config entry.
        sequences: 100           # Distributed across expanded variants in expansion order.
        fixed_elements:
          fixed_element_matrix:
            pairing:
              mode: cross_product    # Budget policy for mining candidates.
            spacer_length: [16, 18]  # Allowed spacer lengths between motif segments.
            upstream_pos: [10, 25]  # Constrain -35 start window.
  runtime:
    max_failed_solutions: 256        # Hard cap on failed solves before abort.
```

```yaml
# LacI/AraC exclusion profile (same config file)
inputs:
  - name: background                        # Identifier for this config entry.
    sampling:
      filters:
        fimo_exclude:
          pwms_input: [lacI_pwm, araC_pwm]  # PWM inputs used by this exclusion filter.
          allow_zero_hit_only: true   # Keep only background with zero LacI/AraC hits.
```

### Step-by-step commands

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/study_constitutive_sigma_panel
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
cd src/dnadesign/densegen/workspaces/study_constitutive_sigma_panel
# Export Cruncher motif artifacts into the DenseGen workspace.
uv run cruncher catalog export-densegen --set 1 --densegen-workspace "$PWD" -c "$(git rev-parse --show-toplevel)/src/dnadesign/cruncher/workspaces/pairwise_laci_arac/configs/config.yaml"
```

### Expected outputs

- `outputs/tables/records.parquet`
- `outputs/meta/events.jsonl`
- `outputs/plots/`
- `outputs/notebooks/densegen_run_overview.py`

### Related docs

- [Generation concept](../concepts/generation.md)
- [Outputs reference](../reference/outputs.md)
- [Workspace catalog](../../workspaces/catalog.md)
