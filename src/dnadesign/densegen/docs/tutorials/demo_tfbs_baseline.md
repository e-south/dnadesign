## DenseGen TFBS baseline tutorial

This tutorial is the shortest end-to-end DenseGen walkthrough and is the best first run for a new operator. Read it when you want to learn the DenseGen lifecycle without PWM mining complexity, and finish with a runnable workspace, plots, and notebook outputs.

### What this tutorial demonstrates
This section states the learning outcomes so you can decide whether this is the right starting point.

- Workspace initialization from a packaged demo.
- Config validation and plan inspection.
- Stage-A plus solve execution for two simple plans.
- Artifact inspection, plotting, notebook generation, and workspace reset.

### Prerequisites
This section ensures your environment is ready before creating a workspace.

```bash
# Install locked Python dependencies for reproducible command behavior.
uv sync --locked

# Confirm DenseGen CLI is available.
uv run dense --help

# Confirm the baseline workspace config validates in the packaged location.
uv run dense validate-config --probe-solver -c src/dnadesign/densegen/workspaces/demo_tfbs_baseline/config.yaml
```

### Key config knobs
This section highlights the highest-signal keys in `src/dnadesign/densegen/workspaces/demo_tfbs_baseline/config.yaml`.

- `densegen.run.id`: Names the run for manifests and logs.
- `densegen.inputs[0].type`: Uses `binding_sites` input from `inputs/sites.csv`.
- `densegen.generation.plan`: Defines `baseline` and `baseline_sigma70` plan behavior.
- `densegen.generation.sequence_length`: Fixes sequence length at `60`.
- `densegen.solver.backend`: Selects solver backend (`CBC` by default).
- `densegen.runtime.max_failed_solutions`: Enforces fail-fast behavior at `0`.
- `densegen.postprocess.pad.mode`: Uses adaptive pad behavior.
- `densegen.output.parquet.path`: Writes canonical records to `outputs/tables/records.parquet`.

### Walkthrough
This section executes the lifecycle in the same order DenseGen runs internally.

#### 1) Create a workspace
This step creates an isolated workspace so you can run and reset safely without modifying the packaged template.

```bash
# Create a new workspace from the packaged TFBS baseline template.
uv run dense workspace init --id tfbs_baseline_trial --from-workspace demo_tfbs_baseline --copy-inputs --output-mode local

# Change into the workspace so all relative output paths resolve locally.
cd src/dnadesign/densegen/workspaces/tfbs_baseline_trial

# Store the config path once for the rest of the tutorial.
CONFIG="$PWD/config.yaml"
```

#### 2) Validate and inspect resolved config
This step confirms strict schema validity and shows which plans and inputs DenseGen will actually run.

```bash
# Validate config structure and probe solver availability.
uv run dense validate-config --probe-solver -c "$CONFIG"

# Inspect resolved Stage-A inputs.
uv run dense inspect inputs -c "$CONFIG"

# Inspect resolved plan and quota settings.
uv run dense inspect plan -c "$CONFIG"
```

#### 3) Run DenseGen
This step executes Stage-A and solve-to-quota from a clean output state.

```bash
# Run from a fresh output directory and skip plot rendering for faster iteration.
uv run dense run --fresh --no-plot -c "$CONFIG"
```

#### 4) Inspect runtime artifacts
This step verifies run health and confirms the records table was materialized.

```bash
# Print event and library summaries to understand run outcomes.
uv run dense inspect run --events --library -c "$CONFIG"

# Verify the canonical records table exists.
ls -la outputs/tables/records.parquet
```

#### 5) Generate plots and notebook
This step produces the default visual outputs and a marimo notebook for interactive review.

```bash
# Render default plots for this workspace.
uv run dense plot -c "$CONFIG"

# Generate the notebook file from run outputs.
uv run dense notebook generate -c "$CONFIG"

# Launch notebook in app mode.
uv run dense notebook run -c "$CONFIG"
```

#### 6) Reset the workspace
This step clears outputs while preserving config and inputs for repeatable reruns.

```bash
# Remove outputs and run state while keeping config and inputs.
uv run dense campaign-reset -c "$CONFIG"
```

### Expected outputs
This section lists the key artifacts you should confirm after a successful run.

- `outputs/tables/records.parquet`
- `outputs/meta/events.jsonl`
- `outputs/meta/run_manifest.json`
- `outputs/plots/placement_map.pdf`
- `outputs/notebooks/densegen_run_overview.py`

### Troubleshooting
This section covers the highest-frequency failures for this baseline tutorial.

- Solver probe fails: rerun `uv run dense validate-config --probe-solver -c "$CONFIG"` and install a supported solver backend.
- Workspace already exists: pick a different `--id` or remove the previous workspace directory.
- Notebook launch fails on remote shell: run `uv run dense notebook run --headless -c "$CONFIG"`.
