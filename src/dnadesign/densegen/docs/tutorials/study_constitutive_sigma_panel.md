## Constitutive sigma70 panel tutorial

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-28


Use this tutorial to build a constitutive σ70 promoter panel. The workflow pairs fixed RNAP -35 and -10 hexamer motifs, then embeds each pair in filtered background sequence.
Background sequence is treated as context only: it is constrained to 40-60% GC and filtered to remove LacI/AraC motif hits.
The -35/-10 motif sets in this workspace follow *Tuning the dynamic range of bacterial promoters regulated by ligand-inducible transcription factors* (DOI: 10.1038/s41467-017-02473-5; source: https://www.nature.com/articles/s41467-017-02473-5).

### Runbook command

Use the workspace runbook sequence from [study_constitutive_sigma_panel/runbook.md](../../workspaces/study_constitutive_sigma_panel/runbook.md). This command runs a clean pass through validation, generation, inspection, and analysis rendering.

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/study_constitutive_sigma_panel
# Run the packaged flow in explicit fresh mode.
./runbook.sh --mode fresh
```

Use `--mode resume` to continue generation, or `--mode analysis` when you only need plots/notebook refresh.

If you want a separate run directory instead of editing this packaged workspace in place, initialize one with `uv run dense workspace init --id <run_id> --from-workspace study_constitutive_sigma_panel --copy-inputs --output-mode local|both`.

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
pixi run dense validate-config --probe-solver -c src/dnadesign/densegen/workspaces/study_constitutive_sigma_panel/config.yaml
```

### Key config sections

```yaml
# src/dnadesign/densegen/workspaces/study_constitutive_sigma_panel/config.yaml
densegen:                                   # DenseGen runtime settings root.
  generation:                               # Sequence-generation controls.
    sequence_length: 60                     # Final sequence length in base pairs.
    expansion:                              # Plan-expansion controls for fixed-element matrices.
      max_plans: 64                         # Max expanded plans; protects against combinatorial growth.
    plan:                                   # List of generation plans before expansion.
      - name: sigma70_panel                 # Plan name shown in outputs and summaries.
        sequences: 100                      # Total target count distributed across expanded variants.
        fixed_elements:                     # Fixed motif constraints for this plan.
          fixed_element_matrix:             # Matrix specification for combinatorial motif pairing.
            pairing:                        # Pairing settings across matrix dimensions.
              mode: cross_product           # Pairing mode; cross_product builds all pair combinations.
            spacer_length: [16, 18]         # Allowed spacer range between -35 and -10 motifs.
            upstream_pos: [10, 25]          # Allowed -35 motif start window in the final sequence.
  runtime:                                  # Runtime stop and retry settings.
    max_failed_solutions: 256               # Hard cap on failed solves before abort.
```

```yaml
# LacI/AraC exclusion profile (same config file)
inputs:                                      # Input definitions used by Stage-A and filters.
  - name: background                         # Input name referenced by plans and filters.
    sampling:                                # Sampling controls for this input.
      filters:                               # Filter chain applied before retention.
        fimo_exclude:                        # FIMO-based exclusion filter.
          pwms_input: [lacI_pwm, araC_pwm]   # PWM inputs checked for exclusion.
          allow_zero_hit_only: true          # True keeps only sequences with zero hits.
```

### Step-by-step commands

Start by pinning the config path used across run commands. This workspace writes local tables to `outputs/tables/` and USR outputs to `outputs/usr_datasets/`.
`dense run` auto-seeds `outputs/usr_datasets/registry.yaml` when it is missing, so no manual registry copy step is required.

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/study_constitutive_sigma_panel
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
cd src/dnadesign/densegen/workspaces/study_constitutive_sigma_panel
# Rebuild plots/notebook from existing run artifacts without regenerating sequences.
./runbook.sh --mode analysis
# Open the generated notebook in marimo app mode.
pixi run dense notebook run -c "$PWD/config.yaml"
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
- [Workspaces directory](../../workspaces/README.md)
