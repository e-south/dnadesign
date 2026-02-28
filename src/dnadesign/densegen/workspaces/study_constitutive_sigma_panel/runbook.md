## study_constitutive_sigma_panel Runbook

**Workspace Path**
- src/dnadesign/densegen/workspaces/study_constitutive_sigma_panel/

**Regulators**
- [lacI, araC] exclusion motifs for background filtering

**Purpose**
- Run the constitutive sigma70 panel with fixed-element expansion and plot/notebook outputs from one workspace.

**σ70 promoter context**
- This workspace composes constitutive σ70 promoters by pairing RNAP -35/-10 site literals from [Tuning the dynamic range of bacterial promoters regulated by ligand-inducible transcription factors](https://www.nature.com/articles/s41467-017-02473-5) (DOI: [10.1038/s41467-017-02473-5](https://doi.org/10.1038/s41467-017-02473-5)).

**Runbook command**

Run this command from the workspace root:

    # Execute the packaged workspace runbook sequence.
    ./runbook.sh

### Step-by-Step Commands

    # Enable strict shell behavior for fail-fast execution.
    set -euo pipefail
    # Pin the workspace config path for repeated CLI calls.
    CONFIG="$PWD/config.yaml"
    # Pin the workspace-local USR registry destination.
    USR_REGISTRY="$PWD/outputs/usr_datasets/registry.yaml"
    # Resolve the repo-level baseline USR registry path.
    ROOT_REGISTRY="$(git rev-parse --show-toplevel)/src/dnadesign/usr/datasets/registry.yaml"

    # Seed a workspace-local USR registry when one is not present.
    if [ ! -f "$USR_REGISTRY" ]; then
      # Create the destination directory for workspace-local USR artifacts.
      mkdir -p "$(dirname "$USR_REGISTRY")"
      # Copy baseline USR registry into the workspace-local path.
      cp "$ROOT_REGISTRY" "$USR_REGISTRY"
    fi

    # Verify FIMO is available before PWM-backed sampling and validation.
    pixi run fimo --version
    # Validate config schema and probe solver availability.
    pixi run dense validate-config --probe-solver -c "$CONFIG"
    # Start a fresh run from a clean output state (sequence generation only).
    # Plot rendering is explicit in the next step for clearer failure isolation.
    pixi run dense run --fresh --no-plot -c "$CONFIG"
    # If running only `dense run`, omit `--no-plot` to auto-render configured plots.
    # pixi run dense run --fresh -c "$CONFIG"
    # Inspect run diagnostics and per-plan library progress.
    pixi run dense inspect run --events --library -c "$CONFIG"
    # Render DenseGen analysis artifacts from current run outputs.
    # `dense plot` is the analysis entry point; static plots always render.
    # This workspace enables plots.video.enabled: true by default, emitting a sampled
    # Stage-B showcase video at outputs/plots/stage_b/all_plans/showcase.mp4.
    # Disable by setting plots.video.enabled: false in config.
    pixi run dense plot -c "$CONFIG"
    # Optional analysis shortcut: render only the Stage-B showcase video artifact.
    # pixi run dense plot --only dense_array_video_showcase -c "$CONFIG"
    # Generate the run-overview marimo notebook artifact.
    pixi run dense notebook generate -c "$CONFIG"
    # Validate the generated notebook before opening or sharing it.
    uv run marimo check "$PWD/outputs/notebooks/densegen_run_overview.py"

### Optional analysis-only mode (existing outputs)

    # Rebuild plots/notebook from existing run artifacts without regenerating sequences.
    ./runbook.sh --analysis-only

### Optional notebook open

    # Launch the generated notebook in marimo app mode.
    pixi run dense notebook run -c "$CONFIG"

### Optional artifact refresh from Cruncher

    # Export Cruncher motif artifacts into this DenseGen workspace.
    uv run cruncher catalog export-densegen --set 1 --densegen-workspace "$PWD" -c "$(git rev-parse --show-toplevel)/src/dnadesign/cruncher/workspaces/pairwise_laci_arac/configs/config.yaml"

### Optional workspace reset

    # Remove run artifacts to return the workspace to a clean state.
    pixi run dense campaign-reset -c "$CONFIG"
