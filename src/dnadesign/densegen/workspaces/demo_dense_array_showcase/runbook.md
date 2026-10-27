## demo_dense_array_showcase Runbook

**Workspace Path**
- src/dnadesign/densegen/workspaces/demo_dense_array_showcase/

**Regulators**
- TF_A, TF_B, TF_C, TF_D, TF_E, TF_F, TF_G, TF_H

**Purpose**
- Run a local, parquet-only DenseGen showcase that packs 40 toy TFBS entries into 100 bp sequences while varying fixed-element pressure across no-anchor, single-anchor, and dual-anchor plans.

**Runbook command**

Run this command from the workspace root:

    # Execute the full runbook flow from a clean output state.
    ./runbook.sh --mode fresh

Use `--mode resume` to continue generation without wiping outputs, or `--mode analysis` to rebuild plots, notebook, and video from existing outputs.

### Step-by-Step Commands

    # Enable strict shell behavior for fail-fast execution.
    set -euo pipefail
    # Pin the workspace config path for repeated CLI calls.
    CONFIG="$PWD/config.yaml"

    # Validate config schema and probe solver availability.
    uv run dense validate-config --probe-solver -c "$CONFIG"
    # Start a fresh run from a clean output state (sequence generation only).
    uv run dense run --fresh --no-plot -c "$CONFIG"
    # Inspect run diagnostics and per-plan library progress.
    uv run dense inspect run --events --library -c "$CONFIG"
    # Render DenseGen analysis artifacts, including the showcase video.
    uv run dense plot -c "$CONFIG"
    # Optional analysis shortcut: render only the Stage-B showcase video artifact.
    # uv run dense plot --only dense_array_showcase_video -c "$CONFIG"
    # Generate the run-overview marimo notebook artifact.
    uv run dense notebook generate -c "$CONFIG"
    # Validate the generated notebook before opening or sharing it.
    uv run marimo check "$PWD/outputs/notebooks/densegen_run_overview.py"

### Expected outputs

- `outputs/tables/records.parquet`
- `outputs/plots/`
- `outputs/plots/stage_b/all_plans/showcase.mp4`
- `outputs/notebooks/densegen_run_overview.py`

### Optional workspace reset

    # Remove run artifacts to return the workspace to a clean state.
    uv run dense campaign-reset -c "$CONFIG"
