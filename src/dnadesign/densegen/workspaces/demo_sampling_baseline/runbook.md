## demo_sampling_baseline Runbook

**Workspace Path**
- src/dnadesign/densegen/workspaces/demo_sampling_baseline/

**Regulators**
- [lexA, cpxR, baeR]

**Purpose**
- Run the sampling baseline from one workspace with FIMO-backed validation, generation, plotting, and notebook output.

**Runbook command**

Run this command from the workspace root:

    ./runbook.sh --mode fresh

Use `--mode resume` to continue generation without wiping outputs, or `--mode analysis` to rebuild plots/notebook only.

### Step-by-Step Commands

    # Enable strict shell behavior for fail-fast execution.
    set -euo pipefail
    # Pin the workspace config path for repeated CLI calls.
    CONFIG="$PWD/config.yaml"
    # dense run auto-seeds outputs/usr_datasets/registry.yaml when missing.

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

### Optional analysis mode (existing outputs)

    # Rebuild plots/notebook from existing run artifacts without regenerating sequences.
    ./runbook.sh --mode analysis

### Optional notebook open

    # Launch the generated notebook in marimo app mode.
    pixi run dense notebook run -c "$CONFIG"

### Optional artifact refresh from Cruncher

    # Export Cruncher motif artifacts into this DenseGen workspace.
    uv run cruncher catalog export-densegen --set 1 --source demo_merged_meme_oops_multitf --densegen-workspace "$PWD" -c "$(git rev-parse --show-toplevel)/src/dnadesign/cruncher/workspaces/demo_multitf/configs/config.yaml"

### Optional workspace reset

    # Remove run artifacts to return the workspace to a clean state.
    pixi run dense campaign-reset -c "$CONFIG"
