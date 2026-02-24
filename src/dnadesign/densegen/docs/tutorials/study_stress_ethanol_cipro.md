## Stress ethanol and ciprofloxacin study tutorial

This tutorial is the end-to-end runbook for the `study_stress_ethanol_cipro` workspace. Read it when you need a stress-response campaign with three base plans expanded across curated sigma70 `-35` variants, higher Stage-A mining budgets, and Notify-ready USR outputs.

### Fast path
For a command-only runbook mirror of this tutorial, use **[`study_stress_ethanol_cipro/runbook.md`](../../workspaces/study_stress_ethanol_cipro/runbook.md)**.

Run this single command from the repository root:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)" && WORKSPACE_ROOT="$REPO_ROOT/src/dnadesign/densegen/workspaces" && WORKSPACE_ID="stress_campaign_trial" && uv run --project "$REPO_ROOT" dense workspace init --id "$WORKSPACE_ID" --root "$WORKSPACE_ROOT" --from-workspace study_stress_ethanol_cipro --copy-inputs --output-mode both && CONFIG="$WORKSPACE_ROOT/$WORKSPACE_ID/config.yaml" && pixi run fimo --version && pixi run dense validate-config --probe-solver -c "$CONFIG" && pixi run dense run --fresh --no-plot -c "$CONFIG" && pixi run dense inspect run --events --library -c "$CONFIG" && pixi run dense plot -c "$CONFIG" && pixi run dense notebook generate -c "$CONFIG"
```

If `WORKSPACE_ID` already exists, choose a new id and rerun the command.

### What this tutorial demonstrates
This section defines the concrete outcomes so you can decide whether this is the right study flow.

- Running the most comprehensive packaged DenseGen study workspace.
- Refreshing workspace-local PWM artifacts from the canonical Cruncher workspace (optional).
- Building high-budget Stage-A pools for LexA, CpxR, BaeR, and constrained background.
- Solving three base plans (`ethanol`, `ciprofloxacin`, `ethanol_ciprofloxacin`) expanded over five sigma70 `-35` variants.
- Verifying dual-sink outputs and the USR event path used by Notify watchers.

### Workspace intent and outcome design
This section explains why this workspace uses matrix expansion and how that maps to intended design outcomes.

- Intent: build a stress-response campaign where regulator constraints define condition identity while sigma70 core tuning controls expression regime.
- Use case: compare ethanol, ciprofloxacin, and combined stress designs under consistent fixed-core expansion semantics.
- Outcome design: keep `-10` fixed at consensus and sweep a curated `-35` set (`f,b,e,d,c`) to preserve diversity while avoiding weak-core heavy libraries.
- Operational meaning: each stress condition compiles into five explicit concrete plans, so downstream plots and records remain condition- and variant-resolved.

### Prerequisites
This section validates your local environment before the longer study run starts.
Run these commands from the repository root so `pixi run dense ...` resolves project tasks correctly.

```bash
# Install locked Python dependencies.
uv sync --locked

# Install pixi-managed tools for FIMO-backed Stage-A mining.
pixi install

# Confirm DenseGen CLI is available.
uv run --project "$(git rev-parse --show-toplevel)" dense --help

# Confirm Cruncher CLI is available.
uv run --project "$(git rev-parse --show-toplevel)" cruncher --help

# Confirm FIMO is available.
pixi run fimo --version

# Validate packaged stress-study config and probe solver availability.
pixi run dense validate-config --probe-solver -c src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/config.yaml
```

### Key config knobs
This section highlights the highest-signal keys in `src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/config.yaml`.

- `densegen.inputs[*].sampling.n_sites`: Sets Stage-A retained pool sizes (`250` PWM sites, `500` background sites).
- `densegen.inputs[*].sampling.mining.budget.candidates`: Sets Stage-A mining effort (`1,000,000` PWM, `5,000,000` background).
- `densegen.inputs.background.sampling.filters.fimo_exclude`: Rejects background candidates that hit study PWMs.
- `densegen.inputs.background.sampling.filters.forbid_kmers`: Uses motif-set rules so sigma exclusion stays coupled to motif-set edits.
- `densegen.motif_sets.sigma70_upstream_35`: Curated strong-to-knee `-35` set (`f,b,e,d,c`).
- `densegen.motif_sets.sigma70_downstream_10`: Fixed consensus `-10` (`TATAAT`).
- `densegen.generation.plan`: Defines three base campaign plans expanded by `fixed_element_matrix`.
- `densegen.generation.plan[*].fixed_elements.fixed_element_matrix`: Expands each base plan over five `-35` variants with `pairing.mode: cross_product`.
- `densegen.generation.plan[*].expanded_name_template: "{base}__sig35={up}"`: Keeps expanded plan labels compact and explicit in records, plots, and notebook selectors.
- `densegen.generation.plan[*].sequences`: Sets per-base-plan sequence milestones that are split evenly across expanded variants.
- `densegen.generation.expansion.max_plans`: Hard guardrail on expanded plan count.
- `densegen.generation.plan[*].regulator_constraints.groups`: Enforces per-condition regulator presence.
- `densegen.generation.sampling.pool_strategy`: Uses Stage-B `subsample` with `library_size: 10`.
- `densegen.generation.sequence_constraints`: Enforces global sigma motif exclusion with fixed-element allowlist.
- `densegen.output.targets`: Uses both `parquet` and `usr` sinks for analysis and Notify readiness.
- `densegen.output.usr.dataset`: Packaged template uses a namespaced dataset id, but `workspace init --output-mode both` rewrites it to your workspace id.
- `densegen.runtime.max_seconds_per_plan`: Caps per-plan solve time at `300` seconds to bound long stalls.
- `densegen.runtime.max_consecutive_failures`: Stops on sustained zero-yield libraries.
- `densegen.runtime.max_failed_solutions_per_target`: Scales rejection budget with quota.
- `densegen.runtime.max_failed_solutions`: Absolute emergency rejection cap per plan.
- `plots.options.placement_map|tfbs_usage.scope: auto` with `max_plans: 12`: defaults Stage-B plotting to grouped output when expanded plan count is large.

### Expansion and quota behavior
This section makes the matrix math explicit for this workspace.

- Matrix expansion treats `sequences` as the base-plan target and enforces an exact divisible split across expanded variants.
- Each base plan uses `fixed_element_matrix` with `pairing.mode: cross_product`.
- Upstream domain size is `5` (`f,b,e,d,c`) and downstream domain size is `1` (`consensus`), so each base plan expands to `5 x 1 = 5` concrete plans.
- Expanded plan count is `3 base plans x 5 = 15`.
- Uniform per-variant quotas are:
  - `ethanol`: `60/5 = 12`
  - `ciprofloxacin`: `60/5 = 12`
  - `ethanol_ciprofloxacin`: `80/5 = 16`
- Aggregate sequence target is `200`.

### Matrix policy in this workspace
This section records the hard behavior guarantees for expansion and runtime planning.

- Expansion is deterministic and resolved before runtime.
- Fail-fast validation rejects invalid motif-set references, invalid selected IDs, invalid pairings, non-divisible sequence splits, and cap overflow.
- Runtime executes resolved concrete plans only; there is no adaptive or fallback plan generation path.

### Walkthrough
This section runs the stress study in the recommended operator order: stage inputs, solve, inspect, and publish outputs.

#### 1) Create an isolated study workspace
This step creates a private workspace so campaign reruns do not mutate the packaged template.

```bash
# Resolve repo root and pin workspace root so paths are deterministic.
REPO_ROOT="$(git rev-parse --show-toplevel)"
WORKSPACE_ROOT="$REPO_ROOT/src/dnadesign/densegen/workspaces"

# Stage a new workspace from the packaged stress-study template with both output sinks enabled.
uv run --project "$REPO_ROOT" dense workspace init --id stress_campaign_trial --root "$WORKSPACE_ROOT" --from-workspace study_stress_ethanol_cipro --copy-inputs --output-mode both

# Cache workspace and config paths for all remaining commands.
WORKSPACE="$WORKSPACE_ROOT/stress_campaign_trial"
CONFIG="$WORKSPACE/config.yaml"
```

#### 2) Refresh PWM artifacts from Cruncher (optional)
This step refreshes workspace-local LexA/CpxR/BaeR artifacts from the canonical Cruncher source.

```bash
# Refresh artifacts into this workspace from the public Cruncher CLI.
uv run --project "$REPO_ROOT" cruncher catalog export-densegen --set 1 --source demo_merged_meme_oops_multitf --densegen-workspace "$WORKSPACE" -c "$REPO_ROOT/src/dnadesign/cruncher/workspaces/demo_multitf/configs/config.yaml"

# Verify refreshed artifact files and manifest.
ls -1 "$WORKSPACE/inputs/motif_artifacts"
```

#### 3) Validate and inspect resolved plan contracts
This step confirms strict schema validity and shows the exact plan/input runtime contract before expensive Stage-A mining.

```bash
# Validate schema and probe solver backend availability.
pixi run dense validate-config --probe-solver -c "$CONFIG"

# Inspect resolved plan names, quotas, and included inputs.
pixi run dense inspect plan -c "$CONFIG"

# Inspect full resolved config for sink and runtime limits.
pixi run dense inspect config -c "$CONFIG"
```

#### 4) Build Stage-A pools from scratch
This step materializes study pools and manifests that Stage-B sampling will draw from.

```bash
# Rebuild Stage-A pools from a clean state.
pixi run dense stage-a build-pool --fresh -c "$CONFIG"

# Inspect Stage-A pool availability by input.
pixi run dense inspect inputs -c "$CONFIG"
```

#### 5) Run solve-to-quota for all expanded stress plans
This step runs Stage-B library sampling and solve loops for all expanded variants in the stress campaign.

```bash
# Execute solve-to-quota and defer plotting until post-run inspection.
pixi run dense run --no-plot -c "$CONFIG"

# Inspect events and library summaries for per-plan progress and rejection causes.
pixi run dense inspect run --events --library -c "$CONFIG"
```

#### 6) Verify Notify-ready USR event outputs
This step confirms the USR event path that Notify watchers must consume.

```bash
# Print the USR events path resolved from this workspace config.
pixi run dense inspect run --usr-events-path -c "$CONFIG"

# Verify USR event log exists at the workspace dataset path.
ls -la "$WORKSPACE/outputs/usr_datasets/stress_campaign_trial/.events.log"
```

#### 7) Render plots and notebook outputs
This step generates review artifacts for run health, usage patterns, and downstream analysis.

```bash
# Render all registered plot types for this stress workspace.
pixi run dense plot -c "$CONFIG"

# Generate notebook from the configured records source.
pixi run dense notebook generate -c "$CONFIG"

# Launch notebook app for interactive inspection.
# If no tab opens in your shell environment, open the printed Notebook URL manually.
pixi run dense notebook run -c "$CONFIG"
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
- No Notify events observed: rerun `pixi run dense inspect run --usr-events-path -c "$CONFIG"` and point Notify at that file.
- Full watcher setup needed: follow **[DenseGen to USR to Notify tutorial](demo_usr_notify.md)**.
