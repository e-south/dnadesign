## Stress ethanol and ciprofloxacin study tutorial

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-28


Use this tutorial to run the largest packaged DenseGen campaign. It combines expanded stress-condition plans, a GUROBI solver backend, and USR-ready outputs around a constitutive σ70 promoter core.
The σ70 core in this workspace is defined by fixed RNAP -35 and -10 hexamer sets from *Tuning the dynamic range of bacterial promoters regulated by ligand-inducible transcription factors* (DOI: 10.1038/s41467-017-02473-5; source: https://www.nature.com/articles/s41467-017-02473-5).

### Runbook command

Use the workspace runbook sequence from [study_stress_ethanol_cipro/runbook.md](../../workspaces/study_stress_ethanol_cipro/runbook.md). This command runs a clean pass through validation, generation, inspection, and analysis rendering.

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro
# Run the packaged flow in explicit fresh mode.
./runbook.sh --mode fresh
```

Use `--mode resume` to continue generation, or `--mode analysis` when you only need plots/notebook refresh.

### Prerequisites

Run these once to install dependencies and verify required CLI tools are available.

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
densegen:                                    # DenseGen runtime settings root.
  output:                                    # Output sink settings.
    targets: [parquet, usr]                  # Output sinks; options include parquet and usr.
  generation:                                # Sequence-generation controls.
    sequence_length: 60                      # Final sequence length in base pairs.
    expansion:                               # Plan-expansion controls for fixed-element matrices.
      max_plans: 64                          # Max expanded plans; protects against combinatorial growth.
    sampling:                                # Stage-B sampling controls.
      library_size: 25                       # Candidate library size sampled per plan.
    plan:                                    # List of base plans before expansion.
      - name: ethanol                        # Plan name shown in outputs and summaries.
        sequences: 300000                    # Base-plan target count before expansion.
      - name: ciprofloxacin                  # Plan name shown in outputs and summaries.
        sequences: 300000                    # Base-plan target count before expansion.
      - name: ethanol_ciprofloxacin          # Plan name shown in outputs and summaries.
        sequences: 400000                    # Base-plan target count before expansion.
  solver:                                    # Dense-array solver settings.
    backend: GUROBI                          # Solver backend; common backends include GUROBI and CBC.
    strategy: iterate                        # Solver strategy; iterate performs repeated bounded passes.
    threads: 12                              # Keep aligned with BU SCC `-pe omp` slots.
  runtime:                                   # Runtime stop and retry settings.
    max_accepted_per_library: 100            # Accepted-sequence cap per Stage-B sampled library.
    max_failed_solutions_per_target: 2.0     # Failed-solve budget scaled by target count.
```

```yaml
# Stage-A campaign mining profile (same config file)
inputs:                                      # Input definitions used by Stage-A and filters.
  - name: lexA_pwm                           # Input name referenced by plans and filters.
    sampling:                                # Sampling controls for this input.
      n_sites: 500                           # Number of retained Stage-A sites for this input.
      mining:                                # Candidate-mining controls before retention.
        batch_size: 5000                     # Candidates evaluated per mining batch.
        budget:                              # Budget policy for Stage-A mining.
          mode: fixed_candidates             # Budget mode; fixed_candidates uses a hard candidate cap.
          candidates: 1000000                # Total candidate budget for this input.
      selection:                             # Candidate selection controls after mining.
        pool:                                # MMR candidate-pool limits.
          max_candidates: 10000              # Cap MMR pool width before final retained-site selection.
  - name: background                         # Input name referenced by plans and filters.
    sampling:                                # Sampling controls for this input.
      n_sites: 500                           # Number of retained Stage-A sites for this input.
      mining:                                # Candidate-mining controls before retention.
        batch_size: 20000                    # Candidates evaluated per mining batch.
        budget:                              # Budget policy for Stage-A mining.
          mode: fixed_candidates             # Budget mode; fixed_candidates uses a hard candidate cap.
          candidates: 1000000                # Total candidate budget for this input.
```

### Step-by-step commands

Start by pinning the config path used across run commands. This workspace writes local tables to `outputs/tables/` and USR outputs to `outputs/usr_datasets/`.
`dense run` auto-seeds `outputs/usr_datasets/registry.yaml` when it is missing, so no manual registry copy step is required.

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro
# Pin config path for repeated CLI calls.
CONFIG="$PWD/config.yaml"
```

#### Mode 1: Core generation run (interactive or OnDemand shell)

Use this mode when running directly in a shell and you want generation-only passes without scheduler wrappers.

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
qsub -P <project> -pe omp 12 -l h_rt=08:00:00 -l mem_per_core=8G -v DENSEGEN_CONFIG="$CONFIG",DENSEGEN_RUN_ARGS='--resume --no-plot' docs/bu-scc/jobs/densegen-cpu.qsub
# Submit a quota-extension pass when additional rows are required.
qsub -P <project> -pe omp 12 -l h_rt=08:00:00 -l mem_per_core=8G -v DENSEGEN_CONFIG="$CONFIG",DENSEGEN_RUN_ARGS='--resume --extend-quota 50000 --no-plot' docs/bu-scc/jobs/densegen-cpu.qsub
```

Queue-fair policy for SCC: if `running_jobs > 3`, avoid burst submits, prefer arrays or `-hold_jid` chains, and do not skip the line.

Million-scale execution model:
- Treat each SCC batch as a contribution pass, not the full campaign.
- For one workspace/run root, keep a single active writer; DenseGen enforces `outputs/meta/run.lock` and concurrent submits on the same workspace exit with a lock-held error.
- For repeated contributions against the same workspace, prefer `-hold_jid` chains over blind parallel submits.
- If you need Stage-A mining/diversity edits, branch to a new workspace (or run root) and separate USR dataset, then merge approved datasets with `uv run usr maintenance merge`.
- Avoid `--fresh` when preserving accumulated rows; `--fresh` clears run artifacts under `outputs/` (while preserving `outputs/notify` and `outputs/logs` scaffolding).

#### Mode 3: Post-run analysis only

Use this mode when generation is complete or paused and you only need plots and notebooks refreshed from existing outputs.

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

### If outputs already exist (analysis mode)

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro
# Rebuild plots/notebook from existing run artifacts without regenerating sequences.
./runbook.sh --mode analysis
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
- `outputs/usr_datasets/densegen/study_stress_ethanol_cipro/.events.log`
- `outputs/plots/`
- `outputs/notebooks/densegen_run_overview.py`

### Related docs

- [DenseGen to USR to Notify tutorial](demo_usr_notify.md)
- [Outputs reference](../reference/outputs.md)
- [Workspaces directory](../../workspaces/README.md)
- [BU SCC Quickstart](../../../../../docs/bu-scc/quickstart.md)
- [BU SCC Batch + Notify runbook](../../../../../docs/bu-scc/batch-notify.md)
- [BU SCC job templates](../../../../../docs/bu-scc/jobs/README.md)
