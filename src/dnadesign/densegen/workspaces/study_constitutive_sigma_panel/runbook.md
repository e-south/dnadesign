## study_constitutive_sigma_panel Runbook

**Workspace Path**
- src/dnadesign/densegen/workspaces/study_constitutive_sigma_panel/

**Regulators**
- [lacI, araC] (strict exclusion motifs against sigma70 panel generation)

**Purpose**
- Run the constitutive sigma70 study end to end with deterministic fixed-element matrix expansion and strict LacI/AraC negative selection.

**Quota Profile**
- `sigma70_panel`: `48` expanded variants from `6x8`, `sequences: 48`, per-variant quota `1`.
- Aggregate: `48` concrete plans, aggregate target `48` (`generation.expansion.max_plans: 64` guardrail on plan-count expansion).
- Expanded plan names are semantic (`__sig35=...__sig10=...`) for clear plan-level records and notebook filtering.
- Stage-B plots default to `scope: auto` (`max_plans: 12`) to avoid per-variant plot sprawl in expanded runs.
- Stage-A diagnostics for this workspace are background-only by design (`generation.plan[].sampling.include_inputs: [background]`), so `stage_a_summary` emits `background_logo` and does not emit PWM mining panels (`pool_tiers`, `yield_bias`, `diversity`).

**Run This Single Command**

Run this single command to do everything below:

    REPO_ROOT="$(git rev-parse --show-toplevel)" && WORKSPACE_ROOT="$REPO_ROOT/src/dnadesign/densegen/workspaces" && WORKSPACE_ID="constitutive_panel_trial" && uv run --project "$REPO_ROOT" dense workspace init --id "$WORKSPACE_ID" --root "$WORKSPACE_ROOT" --from-workspace study_constitutive_sigma_panel --copy-inputs --output-mode both && CONFIG="$WORKSPACE_ROOT/$WORKSPACE_ID/config.yaml" && pixi run fimo --version && pixi run dense validate-config --probe-solver -c "$CONFIG" && pixi run dense run --fresh --no-plot -c "$CONFIG" && pixi run dense inspect run --events --library -c "$CONFIG" && pixi run dense plot -c "$CONFIG" && pixi run dense notebook generate -c "$CONFIG"

If `WORKSPACE_ID` already exists, choose a new id and rerun the command.

### Step-by-Step Commands

    set -euo pipefail
    REPO_ROOT="$(git rev-parse --show-toplevel)"
    WORKSPACE_ROOT="$REPO_ROOT/src/dnadesign/densegen/workspaces"
    WORKSPACE_ID="constitutive_panel_trial"

    # Stage a local workspace so outputs/usr_datasets/registry.yaml is initialized.
    # If this id already exists, choose a new WORKSPACE_ID.
    uv run --project "$REPO_ROOT" dense workspace init --id "$WORKSPACE_ID" --root "$WORKSPACE_ROOT" --from-workspace study_constitutive_sigma_panel --copy-inputs --output-mode both
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

    # Refresh LacI/AraC motif artifacts from Cruncher into this workspace before Stage-A.
    uv run --project "$REPO_ROOT" cruncher catalog export-densegen --set 1 --densegen-workspace "$WORKSPACE" -c "$REPO_ROOT/src/dnadesign/cruncher/workspaces/pairwise_laci_arac/configs/config.yaml"

### Optional workspace reset

    # Remove outputs and run state while keeping config and inputs.
    pixi run dense campaign-reset -c "$CONFIG"
