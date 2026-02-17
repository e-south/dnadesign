## Cruncher to DenseGen PWM handoff

This how-to guide explains the current handoff contract from Cruncher motif artifacts into DenseGen PWM-driven workspaces. Read it when you need a reproducible path from motif discovery to Stage-A pools, Stage-B libraries, and solve-to-quota outputs.

### Current contract summary
This section states the concrete behavior operators should expect from the packaged sampling demo.

The packaged `demo_sampling_baseline` workspace currently uses three PWM artifact inputs (`lexA`, `cpxR`, `baeR`) plus background and runs two plans (`ethanol`, `ciprofloxacin`). The larger `study_stress_ethanol_cipro` workspace is the path for three-plan campaign behavior including `ethanol_ciprofloxacin`.

### Step 1: prepare PWM artifacts
This section clarifies what DenseGen expects from upstream Cruncher outputs.

DenseGen consumes PWM artifacts as explicit per-input files. Do not assume a single aggregated set contract unless your workspace schema explicitly defines one.

### Step 2: initialize a sampling workspace
This section creates a workspace from the packaged baseline and keeps inputs local for reproducibility.

```bash
# Install locked Python dependencies.
uv sync --locked

# Install pixi toolchain when FIMO is needed in Stage-A.
pixi install

# Initialize workspace from packaged sampling baseline.
uv run dense workspace init --id sampling_baseline_trial --from-workspace demo_sampling_baseline --copy-inputs --output-mode both

# Enter the workspace.
cd src/dnadesign/densegen/workspaces/sampling_baseline_trial

# Validate config and solver availability.
uv run dense validate-config --probe-solver
```

### Step 3: build Stage-A pools
This section materializes pool artifacts before solve-to-quota starts.

```bash
# Build Stage-A pools from a clean state.
uv run dense stage-a build-pool --fresh
```

### Step 4: run DenseGen
This section executes Stage-B and solve-to-quota for the sampling baseline plans.

```bash
# Run solve-to-quota and defer plotting.
uv run dense run --no-plot
```

### Step 5: inspect and analyze
This section verifies artifacts and generates analysis surfaces.

```bash
# Inspect run events and library summaries.
uv run dense inspect run --events --library

# Render plots.
uv run dense plot

# Generate notebook.
uv run dense notebook generate
```

### Step 6: optional Notify handoff
This section points to canonical watcher docs instead of repeating full watcher setup.

Use **[DenseGen to USR to Notify tutorial](../tutorials/demo_usr_notify.md)** for end-to-end watcher setup and use **[Notify USR events guide](../../../../../docs/notify/usr-events.md)** for spool/drain operations.
