## DenseGen TFBS baseline tutorial

This tutorial is the smallest end-to-end DenseGen walkthrough. Read it when you want to learn the DenseGen lifecycle without PWM mining complexity, and finish with a runnable workspace, plots, and notebook outputs.

For the stage-by-stage model behind these commands, use **[DenseGen pipeline lifecycle](../concepts/pipeline-lifecycle.md)**.

### Fast path
For a command-only runbook mirror of this tutorial, use **[`demo_tfbs_baseline/runbook.md`](../../workspaces/demo_tfbs_baseline/runbook.md)**.

Run this single command from the repository root:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)" && WORKSPACE_ROOT="$REPO_ROOT/src/dnadesign/densegen/workspaces" && WORKSPACE_ID="tfbs_baseline_trial" && uv run --project "$REPO_ROOT" dense workspace init --id "$WORKSPACE_ID" --root "$WORKSPACE_ROOT" --from-workspace demo_tfbs_baseline --copy-inputs --output-mode local && CONFIG="$WORKSPACE_ROOT/$WORKSPACE_ID/config.yaml" && uv run --project "$REPO_ROOT" dense validate-config --probe-solver -c "$CONFIG" && uv run --project "$REPO_ROOT" dense run --fresh --no-plot -c "$CONFIG" && uv run --project "$REPO_ROOT" dense inspect run --events --library -c "$CONFIG" && uv run --project "$REPO_ROOT" dense plot -c "$CONFIG" && uv run --project "$REPO_ROOT" dense notebook generate -c "$CONFIG"
```

If `WORKSPACE_ID` already exists, choose a new id and rerun the command.

### What this tutorial demonstrates
This section states the learning outcomes so you can decide whether this is the right starting point.

- Workspace initialization from a packaged demo.
- Config validation and plan inspection.
- Stage-A plus solve execution for two simple plans.
- No `fixed_element_matrix` expansion (this tutorial uses explicit non-expanded plans only).
- Artifact inspection, plotting, notebook generation, and workspace reset.

### Prerequisites
This section ensures your environment is ready before creating a workspace.

```bash
# Install locked Python dependencies for reproducible command behavior.
uv sync --locked

# Confirm DenseGen CLI is available.
uv run --project "$(git rev-parse --show-toplevel)" dense --help

# Confirm the baseline workspace config validates in the packaged location.
uv run --project "$(git rev-parse --show-toplevel)" dense validate-config --probe-solver -c src/dnadesign/densegen/workspaces/demo_tfbs_baseline/config.yaml
```

### Key config knobs
This section highlights the highest-signal keys in `src/dnadesign/densegen/workspaces/demo_tfbs_baseline/config.yaml`.

- `densegen.run.id`: Names the run for manifests and logs.
- `densegen.inputs[0].type`: Uses `binding_sites` input from `inputs/sites.csv`.
- `densegen.generation.plan`: Defines `baseline` and `baseline_sigma70` plan behavior.
- `densegen.generation.sequence_length`: Fixes sequence length at `60`.
- `densegen.solver.backend`: Selects solver backend (`CBC` by default).
- `densegen.runtime.max_consecutive_failures`: Stops on sustained zero-yield libraries.
- `densegen.runtime.max_failed_solutions`: Absolute emergency rejection cap (`0` disables).
- `densegen.postprocess.pad.mode`: Uses adaptive pad behavior.
- `densegen.output.parquet.path`: Writes records to `outputs/tables/records.parquet`.

### Walkthrough
This section executes the lifecycle in the same order DenseGen runs internally.

#### 1) Create a workspace
This step creates an isolated workspace so you can run and reset safely without modifying the packaged template.

```bash
# Resolve repo root and pin workspace root so paths are deterministic.
REPO_ROOT="$(git rev-parse --show-toplevel)"
WORKSPACE_ROOT="$REPO_ROOT/src/dnadesign/densegen/workspaces"

# Create a new workspace from the packaged TFBS baseline template.
uv run --project "$REPO_ROOT" dense workspace init --id tfbs_baseline_trial --root "$WORKSPACE_ROOT" --from-workspace demo_tfbs_baseline --copy-inputs --output-mode local

# Change into the workspace so all relative output paths resolve locally.
cd "$WORKSPACE_ROOT/tfbs_baseline_trial"

# Store the config path once for the rest of the tutorial.
CONFIG="$PWD/config.yaml"
```

#### 2) Validate and inspect resolved config
This step confirms strict schema validity and shows which plans and inputs DenseGen will actually run.

```bash
# Validate config structure and probe solver availability.
uv run --project "$REPO_ROOT" dense validate-config --probe-solver -c "$CONFIG"

# Inspect resolved Stage-A inputs.
uv run --project "$REPO_ROOT" dense inspect inputs -c "$CONFIG"

# Inspect resolved plan and quota settings.
uv run --project "$REPO_ROOT" dense inspect plan -c "$CONFIG"
```

#### 3) Run DenseGen
This step executes Stage-A and solve-to-quota from a clean output state.

```bash
# Run from a fresh output directory and skip plot rendering for faster iteration.
uv run --project "$REPO_ROOT" dense run --fresh --no-plot -c "$CONFIG"
```

#### 4) Inspect runtime artifacts
This step verifies run health and confirms the records table was materialized.

```bash
# Print event and library summaries to understand run outcomes.
uv run --project "$REPO_ROOT" dense inspect run --events --library -c "$CONFIG"

# Verify the records table exists.
ls -la outputs/tables/records.parquet
```

#### 5) Generate plots and notebook
This step produces the default visual outputs and a marimo notebook for interactive review.

```bash
# Render all registered plot types; stage_a_summary will be skipped because this workspace has no Stage-A pool artifacts.
uv run --project "$REPO_ROOT" dense plot -c "$CONFIG"

# Generate the notebook file from run outputs.
uv run --project "$REPO_ROOT" dense notebook generate -c "$CONFIG"

# Launch notebook in app mode.
# If no tab opens in your shell environment, open the printed Notebook URL manually.
# Keep this command running while using the notebook; stop with Ctrl+C.
uv run --project "$REPO_ROOT" dense notebook run -c "$CONFIG"

# For subprocess automation, run headless and stop explicitly.
uv run --project "$REPO_ROOT" dense notebook run --headless --port 2718 -c "$CONFIG" &
NOTEBOOK_PID=$!
# ... interact via http://127.0.0.1:2718 ...
kill "$NOTEBOOK_PID"
wait "$NOTEBOOK_PID" || true
```

#### 6) Reset the workspace
This step clears outputs while preserving config and inputs for repeatable reruns.

```bash
# Remove outputs and run state while keeping config and inputs.
uv run --project "$REPO_ROOT" dense campaign-reset -c "$CONFIG"
```

### Expected outputs
This section lists the key artifacts you should confirm after a successful run.

- `outputs/tables/records.parquet`
- `outputs/meta/events.jsonl`
- `outputs/meta/run_manifest.json`
- `outputs/plots/stage_b/<plan>/occupancy.pdf`
- `outputs/plots/stage_b/<plan>/tfbs_usage.pdf`
- `outputs/plots/run_health/*.pdf`
- `outputs/notebooks/densegen_run_overview.py`

### Troubleshooting
This section covers the highest-frequency failures for this baseline tutorial.

- Solver probe fails: rerun `uv run --project "$REPO_ROOT" dense validate-config --probe-solver -c "$CONFIG"` and install a supported solver backend.
- Workspace already exists: pick a different `--id` or remove the previous workspace directory.
- Notebook launch fails on remote shell: run `uv run --project "$REPO_ROOT" dense notebook run --headless -c "$CONFIG"` and open the printed URL manually.
