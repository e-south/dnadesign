## Stress ethanol and ciprofloxacin study tutorial

This tutorial is the end-to-end runbook for the `study_stress_ethanol_cipro` workspace. Read it when you need a stress-response campaign with three plans, higher Stage-A mining budgets, and Notify-ready USR outputs.

### What this tutorial demonstrates
This section defines the concrete outcomes so you can decide whether this is the right study flow.

- Running the most comprehensive packaged DenseGen study workspace.
- Building high-budget Stage-A pools for LexA, CpxR, BaeR, and constrained background.
- Solving three plans (`ethanol`, `ciprofloxacin`, `ethanol_ciprofloxacin`) to quota.
- Verifying dual-sink outputs and the USR event path used by Notify watchers.

### Prerequisites
This section validates your local environment before the longer study run starts.

```bash
# Install locked Python dependencies.
uv sync --locked

# Install pixi-managed tools for FIMO-backed Stage-A mining.
pixi install

# Confirm DenseGen CLI is available.
uv run dense --help

# Confirm FIMO is available.
pixi run fimo --version

# Validate packaged stress-study config and probe solver availability.
uv run dense validate-config --probe-solver -c src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/config.yaml
```

### Key config knobs
This section highlights the highest-signal keys in `src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/config.yaml`.

- `densegen.inputs[*].sampling.n_sites`: Sets Stage-A retained pool sizes (`250` PWM sites, `500` background sites).
- `densegen.inputs[*].sampling.mining.budget.candidates`: Sets Stage-A mining effort (`1,000,000` PWM, `5,000,000` background).
- `densegen.inputs.background.sampling.filters.fimo_exclude`: Rejects background candidates that hit study PWMs.
- `densegen.generation.plan`: Defines three explicit campaign plans and quotas.
- `densegen.generation.plan[*].regulator_constraints.groups`: Enforces per-condition regulator presence.
- `densegen.generation.sampling.pool_strategy`: Uses Stage-B `subsample` with `library_size: 10`.
- `densegen.generation.sequence_constraints`: Enforces global sigma motif exclusion with fixed-element allowlist.
- `densegen.output.targets`: Uses both `parquet` and `usr` sinks for analysis and Notify readiness.
- `densegen.output.usr.dataset`: Packaged template uses a namespaced dataset id, but `workspace init --output-mode both` rewrites it to your workspace id.
- `densegen.runtime.max_seconds_per_plan`: Caps per-plan solve time at `300` seconds to bound long stalls.

### Walkthrough
This section runs the stress study in the recommended operator order: stage inputs, solve, inspect, and publish outputs.

#### 1) Create an isolated study workspace
This step creates a private workspace so campaign reruns do not mutate the packaged template.

```bash
# Resolve repo root and pin workspace root so paths are deterministic.
REPO_ROOT="$(git rev-parse --show-toplevel)"
WORKSPACE_ROOT="$REPO_ROOT/src/dnadesign/densegen/workspaces"

# Stage a new workspace from the packaged stress-study template with both output sinks enabled.
uv run dense workspace init --id stress_campaign_trial --root "$WORKSPACE_ROOT" --from-workspace study_stress_ethanol_cipro --copy-inputs --output-mode both

# Enter the workspace so relative output paths resolve locally.
cd "$WORKSPACE_ROOT/stress_campaign_trial"

# Cache config path for all remaining commands.
CONFIG="$PWD/config.yaml"
```

#### 2) Validate and inspect resolved plan contracts
This step confirms strict schema validity and shows the exact plan/input runtime contract before expensive Stage-A mining.

```bash
# Validate schema and probe solver backend availability.
uv run dense validate-config --probe-solver -c "$CONFIG"

# Inspect resolved plan names, quotas, and included inputs.
uv run dense inspect plan -c "$CONFIG"

# Inspect full resolved config for sink and runtime limits.
uv run dense inspect config -c "$CONFIG"
```

#### 3) Build Stage-A pools from scratch
This step materializes study pools and manifests that Stage-B sampling will draw from.

```bash
# Rebuild Stage-A pools from a clean state.
uv run dense stage-a build-pool --fresh -c "$CONFIG"

# Inspect pool and library summaries for early feasibility signals.
uv run dense inspect run --library -c "$CONFIG"
```

#### 4) Run solve-to-quota for all three plans
This step runs Stage-B library sampling and solve loops for the full stress campaign.

```bash
# Execute solve-to-quota and defer plotting until post-run inspection.
uv run dense run --no-plot -c "$CONFIG"

# Inspect events and library summaries for per-plan progress and rejection causes.
uv run dense inspect run --events --library -c "$CONFIG"
```

#### 5) Verify Notify-ready USR event outputs
This step confirms the USR event path that Notify watchers must consume.

```bash
# Print the USR events path resolved from this workspace config.
uv run dense inspect run --usr-events-path -c "$CONFIG"

# Verify USR event log exists at the workspace dataset path.
ls -la outputs/usr_datasets/stress_campaign_trial/.events.log
```

#### 6) Render plots and notebook outputs
This step generates review artifacts for run health, usage patterns, and downstream analysis.

```bash
# Render all registered plot types for this stress workspace.
uv run dense plot -c "$CONFIG"

# Generate notebook from the configured records source.
uv run dense notebook generate -c "$CONFIG"

# Launch notebook app for interactive inspection.
uv run dense notebook run -c "$CONFIG"
```

### Expected outputs
This section lists the artifacts that confirm the stress study flow completed successfully.

- `outputs/pools/pool_manifest.json`
- `outputs/libraries/`
- `outputs/meta/events.jsonl`
- `outputs/tables/records.parquet`
- `outputs/usr_datasets/stress_campaign_trial/records.parquet`
- `outputs/usr_datasets/stress_campaign_trial/.events.log`
- `outputs/plots/stage_a/*.pdf`
- `outputs/plots/stage_b/<plan>/occupancy.pdf`
- `outputs/plots/stage_b/<plan>/tfbs_usage.pdf`
- `outputs/plots/run_health/*.pdf`
- `outputs/notebooks/densegen_run_overview.py`

### Troubleshooting
This section covers the most common failure modes for this study workspace.

- `fimo` missing: run `pixi run fimo --version` and execute DenseGen via `pixi run dense ...` when needed.
- Stage-A takes too long: reduce `sampling.mining.budget.candidates` for smoke tests, then restore study values.
- Pool exhaustion errors: increase `generation.sampling.library_size` or relax uniqueness constraints.
- No Notify events observed: rerun `uv run dense inspect run --usr-events-path -c "$CONFIG"` and point Notify at that file.
- Full watcher setup needed: follow **[DenseGen to USR to Notify tutorial](demo_usr_notify.md)**.
