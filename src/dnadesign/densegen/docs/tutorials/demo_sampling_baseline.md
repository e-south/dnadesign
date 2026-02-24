## DenseGen sampling baseline tutorial

This tutorial is the DenseGen walkthrough for PWM artifact-driven sampling with Stage-A mining and Stage-B library selection. Read it when you want an end-to-end flow that runs in normal development settings; for the three-plan campaign variant use the **[stress ethanol and ciprofloxacin study tutorial](study_stress_ethanol_cipro.md)**.

### Fast path
For a command-only runbook mirror of this tutorial, use **[`demo_sampling_baseline/runbook.md`](../../workspaces/demo_sampling_baseline/runbook.md)**.

Run this single command from the repository root:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)" && WORKSPACE_ROOT="$REPO_ROOT/src/dnadesign/densegen/workspaces" && WORKSPACE_ID="sampling_baseline_trial" && uv run --project "$REPO_ROOT" dense workspace init --id "$WORKSPACE_ID" --root "$WORKSPACE_ROOT" --from-workspace demo_sampling_baseline --copy-inputs --output-mode both && CONFIG="$WORKSPACE_ROOT/$WORKSPACE_ID/config.yaml" && pixi run fimo --version && pixi run dense validate-config --probe-solver -c "$CONFIG" && pixi run dense run --fresh --no-plot -c "$CONFIG" && pixi run dense inspect run --events --library -c "$CONFIG" && pixi run dense plot -c "$CONFIG" && pixi run dense notebook generate -c "$CONFIG"
```

If `WORKSPACE_ID` already exists, choose a new id and rerun the command.

### What this tutorial demonstrates
This section defines the practical skills you should gain by the end of the run.

- Initializing a dual-sink workspace (`parquet` plus `usr`).
- Refreshing workspace-local PWM artifacts from the canonical Cruncher workspace (optional).
- Building Stage-A pools from three PWM inputs plus background.
- Running Stage-B sampling and solve-to-quota for `ethanol` and `ciprofloxacin` plans.
- No `fixed_element_matrix` expansion (this tutorial uses explicit non-expanded plans with `promoter_constraints`).
- Resuming safely, inspecting artifacts, plotting outputs, and generating a notebook.

### Prerequisites
This section validates local tooling before you spend time on Stage-A mining.
Run these commands from the repository root so `pixi run dense ...` resolves project tasks correctly.

```bash
# Install locked Python dependencies.
uv sync --locked

# Install pixi-managed system tooling when you rely on FIMO.
pixi install

# Confirm DenseGen CLI is available.
uv run --project "$(git rev-parse --show-toplevel)" dense --help

# Confirm Cruncher CLI is available.
uv run --project "$(git rev-parse --show-toplevel)" cruncher --help

# Confirm FIMO is available in your execution path.
pixi run fimo --version

# Confirm packaged sampling config validates and solver probe passes.
pixi run dense validate-config --probe-solver -c src/dnadesign/densegen/workspaces/demo_sampling_baseline/config.yaml
```

### Key config knobs
This section highlights the keys in `src/dnadesign/densegen/workspaces/demo_sampling_baseline/config.yaml` that most affect behavior.

For conceptual tuning guidance, use **[sampling model](../concepts/sampling.md)**.

- `densegen.inputs[*].type`: Uses three `pwm_artifact` inputs and one `background_pool` input.
- `densegen.inputs[*].sampling.n_sites`: Sets retained Stage-A pool size per input.
- `densegen.inputs[*].sampling.mining.budget.candidates`: Sets Stage-A mining effort per input.
- `densegen.generation.sampling.library_size`: Controls Stage-B per-plan library breadth.
- `densegen.generation.plan[*].sampling.include_inputs`: Defines plan-level input composition.
- `densegen.generation.plan[*].fixed_elements.promoter_constraints`: Enforces sigma70 promoter geometry.
- `densegen.generation.sequence_constraints`: Enforces global motif exclusion with explicit allowlist.
- `densegen.output.targets`: Enables dual sink output (`parquet`, `usr`).
- `densegen.output.usr.dataset`: Packaged template uses a namespaced dataset id, but `workspace init --output-mode both` rewrites it to the new workspace id.
- `plots.source`: Chooses the records source used by plotting and notebook flows when dual sinks are enabled.
- `densegen.runtime.max_consecutive_failures`: Stops on sustained zero-yield libraries to prevent endless stalls.
- `densegen.runtime.max_failed_solutions_per_target`: Scales rejection budget with plan quota.
- `densegen.runtime.max_failed_solutions`: Absolute emergency rejection cap per plan.

### Walkthrough
This section follows the same operator sequence you should use in real workspaces.

#### 1) Create a workspace
This step creates an isolated workspace from the packaged sampling demo and preserves local copies of inputs.

```bash
# Resolve repo root and pin workspace root so paths are deterministic.
REPO_ROOT="$(git rev-parse --show-toplevel)"
WORKSPACE_ROOT="$REPO_ROOT/src/dnadesign/densegen/workspaces"

# Initialize workspace from packaged sampling baseline and enable both output sinks.
uv run --project "$REPO_ROOT" dense workspace init --id sampling_baseline_trial --root "$WORKSPACE_ROOT" --from-workspace demo_sampling_baseline --copy-inputs --output-mode both

# Cache workspace and config paths for all remaining commands.
WORKSPACE="$WORKSPACE_ROOT/sampling_baseline_trial"
CONFIG="$WORKSPACE/config.yaml"
```

#### 2) Refresh PWM artifacts from Cruncher (optional)
This step refreshes the three packaged PWM artifacts directly from the canonical Cruncher workspace.

```bash
# Refresh artifacts into this workspace from the public Cruncher CLI.
uv run --project "$REPO_ROOT" cruncher catalog export-densegen --set 1 --source demo_merged_meme_oops_multitf --densegen-workspace "$WORKSPACE" -c "$REPO_ROOT/src/dnadesign/cruncher/workspaces/demo_multitf/configs/config.yaml"

# Verify refreshed artifact files and manifest.
ls -1 "$WORKSPACE/inputs/motif_artifacts"
```

#### 3) Validate and inspect before mining
This step confirms strict config validity and lets you inspect resolved plan/input contracts before expensive work.

```bash
# Validate schema and probe solver backend availability.
pixi run dense validate-config --probe-solver -c "$CONFIG"

# Inspect resolved Stage-A input contract.
pixi run dense inspect inputs -c "$CONFIG"

# Inspect resolved generation plan contract.
pixi run dense inspect plan -c "$CONFIG"

# Inspect resolved full config for sink/source wiring.
pixi run dense inspect config -c "$CONFIG"
```

#### 4) Build Stage-A pools
This step materializes reusable Stage-A pools and writes pool manifests for audit.

```bash
# Build Stage-A pools from scratch and overwrite any prior pool outputs.
pixi run dense stage-a build-pool --fresh -c "$CONFIG"

# Inspect Stage-A pool availability by input.
pixi run dense inspect inputs -c "$CONFIG"
```

#### 5) Run Stage-B plus solve-to-quota
This step runs DenseGen using prepared Stage-A pools and writes both parquet and USR outputs.

```bash
# Execute solve-to-quota without auto-plotting to keep runtime focused on generation.
pixi run dense run --no-plot -c "$CONFIG"
```

#### 6) Resume and extend quota
This step validates resume safety and shows the sanctioned quota-growth flow.

```bash
# Run resume with no config change; this should be a no-op when quotas are already filled.
pixi run dense run --resume --no-plot -c "$CONFIG"

# Resume and extend quotas at runtime for iterative sampling.
pixi run dense run --resume --extend-quota 2 --no-plot -c "$CONFIG"
```

#### 7) Inspect outputs
This step verifies runtime diagnostics, local records, and USR dataset outputs.

```bash
# Print event and library summaries.
pixi run dense inspect run --events --library -c "$CONFIG"

# Confirm local parquet records table exists.
ls -la "$WORKSPACE/outputs/tables/records.parquet"

# Print resolved USR events path for watcher wiring.
pixi run dense inspect run --usr-events-path -c "$CONFIG"
```

#### 8) Render plots and notebook
This step creates presentation artifacts for review and downstream analysis.

```bash
# Render all registered plot types for this workspace.
pixi run dense plot -c "$CONFIG"

# Generate marimo notebook from resolved records source.
pixi run dense notebook generate -c "$CONFIG"

# Run notebook in app mode.
# If no tab opens in your shell environment, open the printed Notebook URL manually.
pixi run dense notebook run -c "$CONFIG"
```

### Expected outputs
This section lists the key files that confirm the sampling baseline ran correctly.

- `outputs/pools/pool_manifest.json`
- `outputs/libraries/`
- `outputs/meta/events.jsonl`
- `outputs/tables/records.parquet`
- `outputs/usr_datasets/sampling_baseline_trial/records.parquet`
- `outputs/usr_datasets/sampling_baseline_trial/.events.log`
- `outputs/plots/stage_a/*.pdf`
- `outputs/plots/stage_b/<plan>/occupancy.pdf`
- `outputs/plots/stage_b/<plan>/tfbs_usage.pdf`
- `outputs/plots/run_health/*.pdf`
- `outputs/notebooks/densegen_run_overview.py`

### Troubleshooting
This section covers the most common operator failures in PWM-heavy flows.

- `fimo` missing: run `pixi run fimo --version` and use `pixi run dense ...` for this tutorial.
- Solver probe fails: rerun `pixi run dense validate-config --probe-solver -c "$CONFIG"` and fix backend install/config.
- Resume rejected after config edits: only quota growth is allowed on resume; otherwise run `--fresh`.
- Notebook source confusion in dual-sink mode: verify `plots.source` in config and rerun `pixi run dense notebook generate -c "$CONFIG"`.
- Notify wiring confusion: use the **[DenseGen to USR to Notify tutorial](demo_usr_notify.md)** and the **[observability and events concept](../concepts/observability_and_events.md)**.
