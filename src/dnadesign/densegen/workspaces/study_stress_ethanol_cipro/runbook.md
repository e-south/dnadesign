## study_stress_ethanol_cipro Runbook

**Workspace Path**
- src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/

**Regulators**
- [lexA, cpxR, baeR]

**Purpose**
- Run the stress study end to end with fixed-element matrix expansion, high-budget Stage-A pools, and dual parquet plus USR outputs.
- Expanded plan names use semantic `__sig35=...` labels and Stage-B plots default to `scope: auto` (`max_plans: 12`) for compact diagnostics.

**Run This Single Command**

Run this single command to do everything below:

    REPO_ROOT="$(git rev-parse --show-toplevel)" && WORKSPACE_ROOT="$REPO_ROOT/src/dnadesign/densegen/workspaces" && WORKSPACE_ID="stress_campaign_trial" && uv run --project "$REPO_ROOT" dense workspace init --id "$WORKSPACE_ID" --root "$WORKSPACE_ROOT" --from-workspace study_stress_ethanol_cipro --copy-inputs --output-mode both && CONFIG="$WORKSPACE_ROOT/$WORKSPACE_ID/config.yaml" && pixi run fimo --version && pixi run dense validate-config --probe-solver -c "$CONFIG" && pixi run dense run --fresh --no-plot -c "$CONFIG" && pixi run dense inspect run --events --library -c "$CONFIG" && pixi run dense plot -c "$CONFIG" && pixi run dense notebook generate -c "$CONFIG"

If `WORKSPACE_ID` already exists, choose a new id and rerun the command.

### Step-by-Step Commands

    set -euo pipefail
    REPO_ROOT="$(git rev-parse --show-toplevel)"
    WORKSPACE_ROOT="$REPO_ROOT/src/dnadesign/densegen/workspaces"
    WORKSPACE_ID="stress_campaign_trial"

    # Stage a local workspace so outputs/usr_datasets/registry.yaml is initialized.
    # If this id already exists, choose a new WORKSPACE_ID.
    uv run --project "$REPO_ROOT" dense workspace init --id "$WORKSPACE_ID" --root "$WORKSPACE_ROOT" --from-workspace study_stress_ethanol_cipro --copy-inputs --output-mode both
    WORKSPACE="$WORKSPACE_ROOT/$WORKSPACE_ID"
    CONFIG="$WORKSPACE/config.yaml"

    # Validate config, FIMO availability, and solver.
    pixi run fimo --version
    pixi run dense validate-config --probe-solver -c "$CONFIG"

    # Run generation from a clean output state.
    pixi run dense run --fresh --no-plot -c "$CONFIG"

    # Inspect runtime events and library summaries.
    pixi run dense inspect run --events --library -c "$CONFIG"

    # Render plots and notebook output.
    pixi run dense plot -c "$CONFIG"
    pixi run dense notebook generate -c "$CONFIG"

### Optional notebook open

    # Launch notebook run mode (opens a browser tab by default).
    # If the requested port already serves this notebook, DenseGen reuses it.
    # If it serves a different notebook, DenseGen starts a fresh server on a free port.
    # Use --reuse-server to force attach to whatever is already running on host/port.
    # If running in a headless shell, open the printed Notebook URL manually.
    pixi run dense notebook run -c "$CONFIG"

### Optional artifact refresh from Cruncher

    # Refresh LexA/CpxR/BaeR motif artifacts from Cruncher into this workspace before Stage-A.
    uv run --project "$REPO_ROOT" cruncher catalog export-densegen --set 1 --source demo_merged_meme_oops_multitf --densegen-workspace "$WORKSPACE" -c "$REPO_ROOT/src/dnadesign/cruncher/workspaces/demo_multitf/configs/config.yaml"

### Optional workspace reset

    # Remove outputs and run state while keeping config and inputs.
    pixi run dense campaign-reset -c "$CONFIG"
