## demo_tfbs_baseline Runbook

**Workspace Path**
- src/dnadesign/densegen/workspaces/demo_tfbs_baseline/

**Regulators**
- [none required]

**Purpose**
- Run the smallest DenseGen baseline from inside one workspace without root-level path exports.

**Runbook command**

Run this command from the workspace root:

    # Execute the packaged workspace runbook sequence.
    ./runbook.sh

### Step-by-Step Commands

    # Enable strict shell behavior for fail-fast execution.
    set -euo pipefail
    # Pin the workspace config path for repeated CLI calls.
    CONFIG="$PWD/config.yaml"

    # Validate config schema and probe solver availability.
    uv run dense validate-config --probe-solver -c "$CONFIG"
    # Start a fresh run from a clean output state (sequence generation only).
    # Plot rendering is explicit in the next step for clearer failure isolation.
    uv run dense run --fresh --no-plot -c "$CONFIG"
    # If running only `dense run`, omit `--no-plot` to auto-render configured plots.
    # uv run dense run --fresh -c "$CONFIG"
    # Inspect run diagnostics and per-plan library progress.
    uv run dense inspect run --events --library -c "$CONFIG"
    # Render DenseGen plots from current run artifacts.
    uv run dense plot -c "$CONFIG"
    # Generate the run-overview marimo notebook artifact.
    uv run dense notebook generate -c "$CONFIG"
    # Validate the generated notebook before opening or sharing it.
    uv run marimo check "$PWD/outputs/notebooks/densegen_run_overview.py"

### Optional analysis-only mode (existing outputs)

    # Rebuild plots/notebook from existing run artifacts without regenerating sequences.
    ./runbook.sh --analysis-only

### Optional notebook open

    # Launch the generated notebook in marimo app mode.
    uv run dense notebook run -c "$CONFIG"

### Optional workspace reset

    # Remove run artifacts to return the workspace to a clean state.
    uv run dense campaign-reset -c "$CONFIG"
