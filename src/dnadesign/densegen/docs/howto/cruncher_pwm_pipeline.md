## Cruncher to DenseGen PWM handoff

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


This how-to guide explains the current handoff contract from Cruncher motif artifacts into DenseGen PWM-driven workspaces. Read it when you need a reproducible path from motif discovery to Stage-A pools, Stage-B libraries, and solve-to-quota outputs.

### Current contract summary
This section states the concrete behavior operators should expect from packaged workspaces.

The packaged `demo_sampling_baseline` workspace currently uses three PWM artifact inputs (`lexA`, `cpxR`, `baeR`) plus background and runs two plans (`ethanol`, `ciprofloxacin`). The larger `study_stress_ethanol_cipro` workspace is the path for three-plan campaign behavior including `ethanol_ciprofloxacin`.
The packaged `study_constitutive_sigma_panel` workspace includes committed `lacI`/`araC` artifacts and uses strict background exclusion (`allow_zero_hit_only=true`) against those PWMs.

### Step 1: Prepare PWM artifacts
This section clarifies what DenseGen expects from upstream Cruncher outputs.

DenseGen consumes PWM artifacts as explicit per-input files. Do not assume a single aggregated set contract unless your workspace schema explicitly defines one.

For exact JSON field requirements, use **[motif artifact JSON contract](../reference/motif_artifacts.md)**. For sampling behavior after ingest, use **[sampling model](../concepts/sampling.md)**.

For packaged workspace refresh from Cruncher:

```bash
# Resolve repository root for stable absolute paths.
REPO_ROOT="$(git rev-parse --show-toplevel)"
# Export Cruncher motif artifacts into the DenseGen workspace.
uv run cruncher catalog export-densegen --set 1 --source demo_merged_meme_oops_multitf --densegen-workspace demo_sampling_baseline -c "$REPO_ROOT/src/dnadesign/cruncher/workspaces/demo_multitf/configs/config.yaml"
# Export Cruncher motif artifacts into the DenseGen workspace.
uv run cruncher catalog export-densegen --set 1 --source demo_merged_meme_oops_multitf --densegen-workspace study_stress_ethanol_cipro -c "$REPO_ROOT/src/dnadesign/cruncher/workspaces/demo_multitf/configs/config.yaml"
# Export Cruncher motif artifacts into the DenseGen workspace.
uv run cruncher catalog export-densegen --set 1 --densegen-workspace study_constitutive_sigma_panel -c "$REPO_ROOT/src/dnadesign/cruncher/workspaces/pairwise_laci_arac/configs/config.yaml"
```

### Step 2: Initialize a sampling workspace
This section creates a workspace from the packaged baseline and keeps inputs local for reproducibility.

```bash
# Install locked Python dependencies.
uv sync --locked

# Install pixi toolchain when FIMO is needed in Stage-A.
pixi install

# Resolve repo root and pin workspace root so paths are deterministic.
REPO_ROOT="$(git rev-parse --show-toplevel)"
# Pin workspace root for deterministic init/output paths.
WORKSPACE_ROOT="$REPO_ROOT/src/dnadesign/densegen/workspaces"

# Initialize workspace from packaged sampling baseline.
uv run dense workspace init --id sampling_baseline_trial --root "$WORKSPACE_ROOT" --from-workspace demo_sampling_baseline --copy-inputs --output-mode both

# Enter the workspace.
cd "$WORKSPACE_ROOT/sampling_baseline_trial"

# Validate config and solver availability.
uv run dense validate-config --probe-solver
```

### Step 3: Build Stage-A pools
This section materializes pool artifacts before solve-to-quota starts.

```bash
# Build Stage-A pools from a clean state.
uv run dense stage-a build-pool --fresh
```

### Step 4: Run DenseGen
This section executes Stage-B and solve-to-quota for the sampling baseline plans.

```bash
# Run solve-to-quota and defer plotting.
uv run dense run --no-plot
```

### Step 5: Inspect and analyze
This section verifies artifacts and generates analysis surfaces.

```bash
# Inspect run events and library summaries.
uv run dense inspect run --events --library

# Render plots.
uv run dense plot

# Generate notebook.
uv run dense notebook generate
```

### Step 6: Optional Notify handoff
This section points to watcher docs instead of repeating full watcher setup.

Use **[DenseGen to USR to Notify tutorial](../tutorials/demo_usr_notify.md)** for end-to-end watcher setup and use **[Notify USR events guide](../../../../../docs/notify/usr-events.md)** for spool/drain operations.
