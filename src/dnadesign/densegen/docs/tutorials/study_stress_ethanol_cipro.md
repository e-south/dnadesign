## Stress ethanol and ciprofloxacin study tutorial

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-28


This tutorial runs the largest packaged DenseGen campaign with expanded stress-condition plans, a GUROBI solver backend, and USR-ready outputs while preserving a constitutive σ70 promoter core.
σ70 -35/-10 literals in this workspace follow [Tuning the dynamic range of bacterial promoters regulated by ligand-inducible transcription factors](https://www.nature.com/articles/s41467-017-02473-5) (DOI: [10.1038/s41467-017-02473-5](https://doi.org/10.1038/s41467-017-02473-5)).

### Runbook command

Use the workspace runbook for the command sequence: [study_stress_ethanol_cipro/runbook.md](../../workspaces/study_stress_ethanol_cipro/runbook.md).

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro
# Execute the packaged workspace runbook sequence.
./runbook.sh
```

### Prerequisites

```bash
# Install locked Python dependencies for reproducible execution.
uv sync --locked
# Install pixi-managed tooling required by this workflow.
pixi install
# Verify FIMO is available before PWM-backed sampling/validation.
pixi run fimo --version
# Validate config schema and probe solver availability.
pixi run dense validate-config --probe-solver -c src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/config.yaml
```

### Key config sections

```yaml
# src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/config.yaml
densegen:
  output:                                 # Output sink configuration.
    targets: [parquet, usr]               # Keep analysis tables and USR events for Notify.
  generation:                             # Generation-stage controls.
    sequence_length: 60                   # Final designed sequence length in base pairs.
    expansion:                            # Plan expansion controls.
      max_plans: 64                       # Upper bound on expanded plan variants.
    sampling:                             # Stage-B library sampling controls.
      library_size: 10                    # Sampled library breadth per plan.
    plan:                                 # Base plans expanded by σ70 variant matrix.
      - name: ethanol                     # Plan identifier.
        sequences: 300000                 # Base-plan quota before 5x expansion.
      - name: ciprofloxacin               # Plan identifier.
        sequences: 300000                 # Base-plan quota before 5x expansion.
      - name: ethanol_ciprofloxacin       # Plan identifier.
        sequences: 400000                 # Base-plan quota before 5x expansion.
  solver:                                 # Dense-array optimizer backend.
    backend: GUROBI                       # Workspace default for SCC-scale runs.
    strategy: iterate                     # Solve strategy.
  runtime:                                # Runtime stall and retry limits.
    max_failed_solutions_per_target: 2.0  # Failed-solve tolerance scaled by target count.
```

```yaml
# Stage-A campaign mining profile (same config file)
inputs:                                   # Stage-A input definitions.
  - name: lexA_pwm                        # Input identifier.
    sampling:                             # Sampling controls for this input.
      n_sites: 250                        # Number of retained sampled sites.
      mining:                             # Candidate mining controls.
        batch_size: 5000                  # Candidate count evaluated per mining batch.
        budget:                           # Candidate mining budget.
          mode: fixed_candidates          # Budget policy for mining candidates.
          candidates: 1000000             # PWM mining effort per regulator.
  - name: background                      # Input identifier.
    sampling:                             # Sampling controls for this input.
      n_sites: 500                        # Number of retained sampled sites.
      mining:                             # Candidate mining controls.
        batch_size: 20000                 # Candidate count evaluated per mining batch.
        budget:                           # Candidate mining budget.
          mode: fixed_candidates          # Budget policy for mining candidates.
          candidates: 5000000             # Background mining budget under exclusion filters.
```

### Step-by-step commands

Start by pinning workspace-local paths used across the run.

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro
# Pin config path for repeated CLI calls.
CONFIG="$PWD/config.yaml"
# Pin workspace-local USR registry destination.
USR_REGISTRY="$PWD/outputs/usr_datasets/registry.yaml"
# Resolve repo-level baseline USR registry path.
ROOT_REGISTRY="$(git rev-parse --show-toplevel)/src/dnadesign/usr/datasets/registry.yaml"
```

Seed the workspace-local USR registry once so USR output writes stay deterministic.

```bash
# Seed a workspace-local USR registry when one is not present.
if [ ! -f "$USR_REGISTRY" ]; then
  # Create the target directory if it does not already exist.
  mkdir -p "$(dirname "$USR_REGISTRY")"
  # Copy baseline artifacts into the workspace-local location.
  cp "$ROOT_REGISTRY" "$USR_REGISTRY"
fi
```

#### Mode 1: Core generation run (interactive or OnDemand shell)

Use this mode when you are running in a shell directly (local terminal or interactive remote shell) and want generation-only passes without scheduler wrappers.

```bash
# Verify FIMO is available before PWM-backed sampling/validation.
pixi run fimo --version
# Validate config schema and probe solver availability.
pixi run dense validate-config --probe-solver -c "$CONFIG"
# Start a fresh run when beginning a new campaign branch.
pixi run dense run --fresh --no-plot -c "$CONFIG"
# Resume without wiping outputs on subsequent passes.
pixi run dense run --resume --no-plot -c "$CONFIG"
# Increase total target by a bounded amount without editing config.yaml.
pixi run dense run --resume --extend-quota 50000 --no-plot -c "$CONFIG"
# Inspect run diagnostics and current per-plan progress toward quota.
pixi run dense inspect run --events --library -c "$CONFIG"
```

#### Mode 2: BU SCC batch loop (target quota)

Use this mode for asynchronous SCC execution. The workspace default target is 1,000,000 total sequences (after expansion), so multiple submissions are expected.

```bash
# Check scheduler pressure before new submissions.
qstat -u "$USER"
# Summarize running, queued, and Eqw jobs for submit gating.
qstat -u "$USER" | awk '$1 ~ /^[0-9]+$/ { running += ($5 ~ /r/); queued += ($5 ~ /q/); eqw += ($5 ~ /Eqw/) } END { printf "running_jobs=%d queued_jobs=%d eqw_jobs=%d\n", running, queued, eqw }'
# Submit generation-only DenseGen batch against this workspace config.
qsub -P <project> -v DENSEGEN_CONFIG="$CONFIG",DENSEGEN_RUN_ARGS='--resume --no-plot' docs/bu-scc/jobs/densegen-cpu.qsub
# Submit a quota-extension pass when additional rows are required.
qsub -P <project> -v DENSEGEN_CONFIG="$CONFIG",DENSEGEN_RUN_ARGS='--resume --extend-quota 50000 --no-plot' docs/bu-scc/jobs/densegen-cpu.qsub
```

Queue-fair policy for SCC: if `running_jobs > 3`, avoid burst submits, prefer arrays or `-hold_jid` chains, and do not skip the line.

Million-scale execution model:
- Treat each SCC batch as a contribution pass, not the full campaign.
- For one workspace/run root, keep a single active writer; DenseGen enforces `outputs/meta/run.lock` and concurrent submits on the same workspace exit with a lock-held error.
- For repeated contributions against the same workspace, prefer `-hold_jid` chains over blind parallel submits.
- If you need Stage-A mining/diversity edits, branch to a new workspace (or run root) and separate USR dataset, then merge approved datasets with `uv run usr maintenance merge`.
- Avoid `--fresh` when preserving accumulated rows; `--fresh` clears `outputs/` (except `outputs/notify`).

#### Mode 3: Post-run analysis only

Use this mode when sequence generation is complete (or paused) and you only need plots/notebooks refreshed from existing outputs.

```bash
# Render DenseGen analysis artifacts from current run outputs.
pixi run dense plot -c "$CONFIG"
# Optional analysis shortcut: render only the Stage-B showcase video artifact.
# pixi run dense plot --only dense_array_video_showcase -c "$CONFIG"
# Generate the run-overview marimo notebook artifact.
pixi run dense notebook generate -c "$CONFIG"
# Run notebook validation before opening or sharing it.
uv run marimo check "$PWD/outputs/notebooks/densegen_run_overview.py"
```

Resume safety contract:
- In-place resume accepts quota-only config changes (for example `sequences` increases).
- Changes to Stage-A mining/selection knobs, sequence length, fixed elements, or other non-quota config keys are blocked in-place with `Config changed beyond plan quotas.`.
- For those broader changes, start a new run root (or run `--fresh`) to avoid mixing incompatible state.

Durability knobs for interruption tolerance:
- `densegen.runtime.checkpoint_every` controls how often run state and sink buffers are checkpointed.
- `densegen.output.parquet.chunk_size` and `densegen.output.usr.chunk_size` control buffered write size per flush.
- Lower values reduce in-memory exposure on hard interruption, with higher I/O overhead.

### If outputs already exist (analysis-only)

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro
# Rebuild plots/notebook from existing run artifacts without regenerating sequences.
./runbook.sh --analysis-only
# Open the generated notebook in marimo app mode.
pixi run dense notebook run -c "$PWD/config.yaml"
```

### Optional artifact refresh from Cruncher

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro
# Export Cruncher motif artifacts into the DenseGen workspace.
uv run cruncher catalog export-densegen --set 1 --source demo_merged_meme_oops_multitf --densegen-workspace "$PWD" -c "$(git rev-parse --show-toplevel)/src/dnadesign/cruncher/workspaces/demo_multitf/configs/config.yaml"
```

### Expected outputs

- `outputs/tables/records.parquet`
- `outputs/usr_datasets/study_stress_ethanol_cipro/.events.log`
- `outputs/plots/`
- `outputs/notebooks/densegen_run_overview.py`

### Related docs

- [DenseGen to USR to Notify tutorial](demo_usr_notify.md)
- [Outputs reference](../reference/outputs.md)
- [Workspace catalog](../../workspaces/catalog.md)
- [BU SCC Quickstart](../../../../../docs/bu-scc/quickstart.md)
- [BU SCC Batch + Notify runbook](../../../../../docs/bu-scc/batch-notify.md)
- [BU SCC job templates](../../../../../docs/bu-scc/jobs/README.md)
