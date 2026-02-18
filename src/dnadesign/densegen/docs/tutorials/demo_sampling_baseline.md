## DenseGen sampling baseline tutorial

This tutorial is the DenseGen walkthrough for PWM artifact-driven sampling with Stage-A mining and Stage-B library selection. Read it when you want an end-to-end flow that runs in normal development settings; for the three-plan campaign variant use the **[stress ethanol and ciprofloxacin study tutorial](study_stress_ethanol_cipro.md)**.

### What this tutorial demonstrates
This section defines the practical skills you should gain by the end of the run.

- Initializing a dual-sink workspace (`parquet` plus `usr`).
- Building Stage-A pools from three PWM inputs plus background.
- Running Stage-B sampling and solve-to-quota for `ethanol` and `ciprofloxacin` plans.
- Resuming safely, inspecting artifacts, plotting outputs, and generating a notebook.

### Prerequisites
This section validates local tooling before you spend time on Stage-A mining.

```bash
# Install locked Python dependencies.
uv sync --locked

# Install pixi-managed system tooling when you rely on FIMO.
pixi install

# Confirm DenseGen CLI is available.
uv run dense --help

# Confirm FIMO is available in your execution path.
pixi run fimo --version

# Confirm packaged sampling config validates and solver probe passes.
uv run dense validate-config --probe-solver -c src/dnadesign/densegen/workspaces/demo_sampling_baseline/config.yaml
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
- `densegen.runtime.max_failed_solutions`: Preserves fail-fast contract with no silent fallback.

### Walkthrough
This section follows the same operator sequence you should use in real workspaces.

#### 1) Create a workspace
This step creates an isolated workspace from the packaged sampling demo and preserves local copies of inputs.

```bash
# Resolve repo root and pin workspace root so paths are deterministic.
REPO_ROOT="$(git rev-parse --show-toplevel)"
WORKSPACE_ROOT="$REPO_ROOT/src/dnadesign/densegen/workspaces"

# Initialize workspace from packaged sampling baseline and enable both output sinks.
uv run dense workspace init --id sampling_baseline_trial --root "$WORKSPACE_ROOT" --from-workspace demo_sampling_baseline --copy-inputs --output-mode both

# Enter the workspace so relative output paths resolve correctly.
cd "$WORKSPACE_ROOT/sampling_baseline_trial"

# Cache config path for all remaining commands.
CONFIG="$PWD/config.yaml"
```

#### 2) Validate and inspect before mining
This step confirms strict config validity and lets you inspect resolved plan/input contracts before expensive work.

```bash
# Validate schema and probe solver backend availability.
uv run dense validate-config --probe-solver -c "$CONFIG"

# Inspect resolved Stage-A input contract.
uv run dense inspect inputs -c "$CONFIG"

# Inspect resolved generation plan contract.
uv run dense inspect plan -c "$CONFIG"

# Inspect resolved full config for sink/source wiring.
uv run dense inspect config -c "$CONFIG"
```

#### 3) Build Stage-A pools
This step materializes reusable Stage-A pools and writes pool manifests for audit.

```bash
# Build Stage-A pools from scratch and overwrite any prior pool outputs.
uv run dense stage-a build-pool --fresh -c "$CONFIG"

# Inspect Stage-A pool manifest paths and counts.
uv run dense inspect run --library -c "$CONFIG"
```

#### 4) Run Stage-B plus solve-to-quota
This step runs DenseGen using prepared Stage-A pools and writes both parquet and USR outputs.

```bash
# Execute solve-to-quota without auto-plotting to keep runtime focused on generation.
uv run dense run --no-plot -c "$CONFIG"
```

#### 5) Resume and extend quota
This step validates resume safety and shows the sanctioned quota-growth flow.

```bash
# Run resume with no config change; this should be a no-op when quotas are already filled.
uv run dense run --resume --no-plot -c "$CONFIG"

# Resume and extend quotas at runtime for iterative sampling.
uv run dense run --resume --extend-quota 2 --no-plot -c "$CONFIG"
```

#### 6) Inspect outputs
This step verifies runtime diagnostics, local records, and USR dataset outputs.

```bash
# Print event and library summaries.
uv run dense inspect run --events --library -c "$CONFIG"

# Confirm local parquet records table exists.
ls -la outputs/tables/records.parquet

# Print resolved USR events path for watcher wiring.
uv run dense inspect run --usr-events-path -c "$CONFIG"
```

#### 7) Render plots and notebook
This step creates presentation artifacts for review and downstream analysis.

```bash
# Render all registered plot types for this workspace.
uv run dense plot -c "$CONFIG"

# Generate marimo notebook from resolved records source.
uv run dense notebook generate -c "$CONFIG"

# Run notebook in app mode.
uv run dense notebook run -c "$CONFIG"
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

- `fimo` missing: run `pixi run fimo --version` and use `pixi run dense ...` when FIMO is only available in pixi.
- Solver probe fails: rerun `uv run dense validate-config --probe-solver -c "$CONFIG"` and fix backend install/config.
- Resume rejected after config edits: only quota growth is allowed on resume; otherwise run `--fresh`.
- Notebook source confusion in dual-sink mode: verify `plots.source` in config and rerun `dense notebook generate`.
- Notify wiring confusion: use the **[DenseGen to USR to Notify tutorial](demo_usr_notify.md)** and the **[observability and events concept](../concepts/observability_and_events.md)**.
