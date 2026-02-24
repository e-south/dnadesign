## demo_tfbs_baseline Runbook

**Workspace Path**
- src/dnadesign/densegen/workspaces/demo_tfbs_baseline/

**Regulators**
- [TF_A, TF_B, TF_C]

**Purpose**
- Run the smallest DenseGen flow end to end to validate config, solve behavior, and artifact generation.

**Run This Single Command**

Run this single command to do everything below:

    REPO_ROOT="$(git rev-parse --show-toplevel)" && WORKSPACE_ROOT="$REPO_ROOT/src/dnadesign/densegen/workspaces" && WORKSPACE_ID="tfbs_baseline_trial" && uv run --project "$REPO_ROOT" dense workspace init --id "$WORKSPACE_ID" --root "$WORKSPACE_ROOT" --from-workspace demo_tfbs_baseline --copy-inputs --output-mode local && CONFIG="$WORKSPACE_ROOT/$WORKSPACE_ID/config.yaml" && uv run --project "$REPO_ROOT" dense validate-config --probe-solver -c "$CONFIG" && uv run --project "$REPO_ROOT" dense run --fresh --no-plot -c "$CONFIG" && uv run --project "$REPO_ROOT" dense inspect run --events --library -c "$CONFIG" && uv run --project "$REPO_ROOT" dense plot -c "$CONFIG" && uv run --project "$REPO_ROOT" dense notebook generate -c "$CONFIG"

If `WORKSPACE_ID` already exists, choose a new id and rerun the command.

### Step-by-Step Commands

    set -euo pipefail
    REPO_ROOT="$(git rev-parse --show-toplevel)"
    WORKSPACE_ROOT="$REPO_ROOT/src/dnadesign/densegen/workspaces"
    WORKSPACE_ID="tfbs_baseline_trial"

    # Stage a local workspace from the packaged template.
    # If this id already exists, choose a new WORKSPACE_ID.
    uv run --project "$REPO_ROOT" dense workspace init --id "$WORKSPACE_ID" --root "$WORKSPACE_ROOT" --from-workspace demo_tfbs_baseline --copy-inputs --output-mode local
    WORKSPACE="$WORKSPACE_ROOT/$WORKSPACE_ID"
    CONFIG="$WORKSPACE/config.yaml"

    # Validate schema and solver availability.
    uv run --project "$REPO_ROOT" dense validate-config --probe-solver -c "$CONFIG"

    # Run from a clean output state.
    uv run --project "$REPO_ROOT" dense run --fresh --no-plot -c "$CONFIG"

    # Inspect runtime events and library summaries.
    uv run --project "$REPO_ROOT" dense inspect run --events --library -c "$CONFIG"

    # Render plots and notebook outputs.
    uv run --project "$REPO_ROOT" dense plot -c "$CONFIG"
    uv run --project "$REPO_ROOT" dense notebook generate -c "$CONFIG"

### Optional notebook open

    # Launch notebook run mode (opens a browser tab by default).
    # If the requested port already serves this notebook, DenseGen reuses it.
    # If it serves a different notebook, DenseGen starts a fresh server on a free port.
    # Use --reuse-server to force attach to whatever is already running on host/port.
    # Keep this command running while you use the notebook; stop with Ctrl+C.
    # If running in a headless shell, open the printed Notebook URL manually.
    uv run --project "$REPO_ROOT" dense notebook run -c "$CONFIG"

    # Codex/subprocess flow: start headless notebook server, then stop it explicitly.
    uv run --project "$REPO_ROOT" dense notebook run --headless --port 2718 -c "$CONFIG" &
    NOTEBOOK_PID=$!
    # ... interact via http://127.0.0.1:2718 ...
    kill "$NOTEBOOK_PID"
    wait "$NOTEBOOK_PID" || true

### Optional workspace reset

    # Remove outputs and run state while keeping config and inputs.
    uv run --project "$REPO_ROOT" dense campaign-reset -c "$CONFIG"
