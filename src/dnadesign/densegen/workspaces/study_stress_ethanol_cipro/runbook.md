## study_stress_ethanol_cipro Runbook

**Workspace Path**
- src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/

**Regulators**
- [lexA, cpxR, baeR]

**Purpose**
- Run the stress campaign workspace with dual-sink outputs, expanded plans, GUROBI solver defaults, and workspace-local plot/notebook generation.
- Keep DenseGen accumulation in the shared USR root
  `src/dnadesign/usr/datasets/` so the study record and downstream tools read
  the same producer dataset directly.

**σ70 promoter context**
- This workspace keeps a constitutive σ70 promoter core and uses RNAP -35 and -10 hexamer sets from *Tuning the dynamic range of bacterial promoters regulated by ligand-inducible transcription factors* (DOI: 10.1038/s41467-017-02473-5; source: https://www.nature.com/articles/s41467-017-02473-5).

**Runbook command**

Run this command from the workspace root:

    # Execute the full runbook flow from a clean output state.
    ./runbook.sh --mode fresh

Use `--mode resume` to continue generation without wiping outputs, or `--mode analysis` to rebuild plots/notebook only.

### Step-by-Step Commands

    # Enable strict shell behavior for fail-fast execution.
    set -euo pipefail
    # Pin the workspace config path for repeated CLI calls.
    CONFIG="$PWD/config.yaml"
    # dense run auto-seeds the configured shared USR root registry when missing.

    # Verify FIMO is available before PWM-backed sampling and validation.
    pixi run fimo --version
    # Validate config schema and probe solver availability.
    pixi run dense validate-config --probe-solver -c "$CONFIG"
    # Start a fresh run from a clean output state (sequence generation only).
    # Plot rendering is explicit in the next step for clearer failure isolation.
    pixi run dense run --fresh --no-plot -c "$CONFIG"
    # Resume generation in-place for iterative quota accumulation.
    pixi run dense run --resume --no-plot -c "$CONFIG"
    # Increase total quota target without editing config.yaml.
    pixi run dense run --resume --extend-quota 50000 --no-plot -c "$CONFIG"
    # If running only `dense run`, omit `--no-plot` to auto-render configured plots.
    # pixi run dense run --fresh -c "$CONFIG"
    # Inspect run diagnostics and per-plan library progress.
    pixi run dense inspect run --events --library -c "$CONFIG"
    # Render DenseGen analysis artifacts from current run outputs.
    # The default catalog now includes dataset-native read-only analysis plots
    # over the selected records source plus the core local diagnostics.
    pixi run dense plot -c "$CONFIG"
    # Optional analysis shortcut: re-render only the Stage-B showcase video artifact.
    # pixi run dense plot --only dense_array_showcase_video -c "$CONFIG"
    # Generate the run-overview marimo notebook artifact.
    pixi run dense notebook generate -c "$CONFIG"
    # Validate the generated notebook before opening or sharing it.
    uv run marimo check "$PWD/outputs/notebooks/densegen_run_overview.py"

### Read-only local DenseGen analysis

These commands read the shared `densegen/study_stress_ethanol_cipro` dataset
through the existing workspace config and only write local plot/notebook
artifacts under `outputs/`. The workspace resolves `output.usr.root` against
the git common repo root so the same commands work from a normal checkout and
from an isolated worktree.

    # Render the default read-only analysis catalog from the shared DenseGen source dataset.
    uv run dense plot -c "$CONFIG"
    # Generate a marimo notebook from the shared DenseGen source dataset.
    uv run dense notebook generate --force -c "$CONFIG"
    # Validate the generated notebook artifact.
    uv run marimo check "$PWD/outputs/notebooks/densegen_run_overview.py"
    # Launch the notebook in marimo app mode.
    uv run dense notebook run -c "$CONFIG"

The default plot catalog for this workspace now renders the full
notebook-visible plot surface required for notebook launch. The happy path is
that `dense plot` materializes every plot shown in the generated notebook,
including the drilldown panels and the Stage-B showcase video.

### Mode B: BU SCC batch loop (generation only)

    # Check current scheduler pressure before submitting additional jobs.
    qstat -u "$USER"
    # Summarize running, queued, and Eqw jobs for submit gating.
    qstat -u "$USER" | awk '
      $1 ~ /^[0-9]+$/ {
        running += ($5 ~ /r/)
        queued += ($5 ~ /q/)
        eqw += ($5 ~ /Eqw/)
      }
      END { printf "running_jobs=%d queued_jobs=%d eqw_jobs=%d\n", running, queued, eqw }
    '
    # Submit generation-only batch run against this workspace config.
    qsub -P <project> \
      -pe omp 12 \
      -l h_rt=08:00:00 \
      -l mem_per_core=8G \
      -v DENSEGEN_CONFIG="$CONFIG",DENSEGEN_RUN_ARGS='--resume --no-plot' \
      docs/bu-scc/jobs/densegen-cpu.qsub
    # Submit an extension pass when additional quota is required.
    qsub -P <project> \
      -pe omp 12 \
      -l h_rt=08:00:00 \
      -l mem_per_core=8G \
      -v DENSEGEN_CONFIG="$CONFIG",DENSEGEN_RUN_ARGS='--resume --extend-quota 50000 --no-plot' \
      docs/bu-scc/jobs/densegen-cpu.qsub

Queue contract:
- If `running_jobs > 3`, confirm before adding more jobs and prefer arrays or `-hold_jid` chains.
- Do not skip the queue line with bypass-style flags.
- Keep `densegen.solver.threads` aligned with `-pe omp` slots (this workspace uses `12`).

### Optional analysis mode (existing outputs)

    Mode C: post-run analysis only.

        # Rebuild plots/notebook from existing run artifacts without regenerating sequences.
        # Records-only state is acceptable here: the shared runbook continues when
        # finalized run metadata is absent and still refreshes recoverable diagnostics and Stage-B plots.
        ./runbook.sh --mode analysis

### Optional notebook open

    # Launch the generated notebook in marimo app mode.
    pixi run dense notebook run -c "$CONFIG"

### Optional artifact refresh from Cruncher

    # Export Cruncher motif artifacts into this DenseGen workspace.
    uv run cruncher catalog export-densegen --set 1 --source demo_merged_meme_oops_multitf --densegen-workspace "$PWD" -c "$(git rev-parse --show-toplevel)/src/dnadesign/cruncher/workspaces/demo_multitf/configs/config.yaml"

### Optional workspace reset

    # Remove run artifacts to return the workspace to a clean state.
    pixi run dense campaign-reset -c "$CONFIG"
