## Constitutive sigma panel study tutorial

This tutorial exercises the `study_constitutive_sigma_panel` workspace, which is the DenseGen example for constitutive promoter core-element combinatorics. Read it when you need to understand plan-template expansion, promoter-matrix placement, and strict global motif exclusion behavior.

### What this tutorial demonstrates
This section states the capabilities this study adds beyond baseline demo workspaces.

- Config-time expansion of `generation.plan_templates` into concrete plans.
- Use of `motif_sets` plus `fixed_elements.promoter_matrix` for combinatorial promoter cores.
- Global strand-aware motif exclusion with explicit allowlist exceptions.
- Study-scale dual-sink outputs with notebook and plot review.

### Prerequisites
This section validates the core dependencies before running a larger study workflow.

```bash
# Install locked Python dependencies.
uv sync --locked

# Confirm DenseGen CLI is available.
uv run dense --help

# Confirm constitutive study config validates and solver probe passes.
uv run dense validate-config --probe-solver -c src/dnadesign/densegen/workspaces/study_constitutive_sigma_panel/config.yaml
```

### Key config knobs
This section highlights the schema surfaces this study uses that the baseline demos do not.

- `densegen.motif_sets`: Defines reusable sigma motif libraries (`-35` and `-10` hexamers).
- `densegen.generation.plan_templates`: Defines templated plans that expand into concrete plan instances.
- `densegen.generation.plan_template_max_expanded_plans`: Hard cap for expansion size to prevent accidental combinatoric blowups.
- `densegen.generation.plan_template_max_total_quota`: Hard cap for aggregate quota across expanded plans.
- `densegen.generation.plan_templates[*].fixed_elements.promoter_matrix`: Declares upstream/downstream motif set pairing and spacer geometry.
- `densegen.generation.sequence_constraints.forbid_kmers`: Declares global forbidden motif set scanning on both strands.
- `densegen.generation.sequence_constraints.allowlist`: Declares explicit fixed-element exceptions for intended placements.
- `densegen.inputs[0].sampling.filters.forbid_kmers`: Applies Stage-A background prefiltering for sigma motifs.
- `densegen.output.targets`: Uses both parquet and USR sinks for analysis and watcher compatibility.
- `plots.default`: Includes placement and run-health plots tuned for study review.

### Walkthrough
This section runs the study workspace in the intended order: expansion validation, pool build, run, and analysis.

#### 1) Create a study workspace
This step copies the packaged study into a private workspace for safe iteration.

```bash
# Initialize a new workspace from the constitutive study template.
uv run dense workspace init --id constitutive_panel_trial --from-workspace study_constitutive_sigma_panel --copy-inputs --output-mode both

# Enter the workspace.
cd src/dnadesign/densegen/workspaces/constitutive_panel_trial

# Cache config path for later commands.
CONFIG="$PWD/config.yaml"
```

#### 2) Validate and inspect expanded plans
This step confirms template expansion behavior before any expensive generation.

```bash
# Validate strict schema and probe solver.
uv run dense validate-config --probe-solver -c "$CONFIG"

# Inspect resolved generation plans after template expansion.
uv run dense inspect plan -c "$CONFIG"

# Inspect full resolved config to confirm motif_sets and sequence constraints.
uv run dense inspect config -c "$CONFIG"
```

#### 3) Build Stage-A background pool
This step builds the constrained background pool used by all expanded constitutive plans.

```bash
# Build Stage-A pools from scratch.
uv run dense stage-a build-pool --fresh -c "$CONFIG"

# Inspect pool and library summaries for early feasibility signals.
uv run dense inspect run --library -c "$CONFIG"
```

#### 4) Run DenseGen and inspect run health
This step executes solve-to-quota for expanded plans with strict motif constraints.

```bash
# Run study generation and defer plot rendering until after inspection.
uv run dense run --no-plot -c "$CONFIG"

# Inspect event stream and library summaries for rejection reasons and progress.
uv run dense inspect run --events --library -c "$CONFIG"
```

#### 5) Render analysis artifacts
This step builds review artifacts for promoter placement geometry and run-level quality checks.

```bash
# Render default study plots.
uv run dense plot -c "$CONFIG"

# Generate the notebook for run analysis.
uv run dense notebook generate -c "$CONFIG"

# Launch notebook in app mode.
uv run dense notebook run -c "$CONFIG"
```

### Expected outputs
This section lists the artifacts that confirm the constitutive study contract executed as intended.

- `outputs/pools/pool_manifest.json`
- `outputs/meta/effective_config.yaml` (includes expanded plans)
- `outputs/meta/events.jsonl`
- `outputs/tables/records.parquet`
- `outputs/usr_datasets/densegen/study_constitutive_sigma_panel/records.parquet`
- `outputs/plots/placement_map.pdf`
- `outputs/plots/run_health.pdf`
- `outputs/notebooks/densegen_run_overview.py`

### Troubleshooting
This section covers the most common constitutive-panel footguns.

- Expansion fails with cap error: reduce motif-set size or adjust `plan_template_max_expanded_plans` intentionally.
- Runtime too long: reduce `total_quota` per template for smoke tests before full study runs.
- Constraint rejection spikes: inspect `events.jsonl` for sequence-constraint failures and confirm allowlist coordinates are correct.
- Notebook seems to show wrong source: verify `plots.source` when dual sinks are enabled.
