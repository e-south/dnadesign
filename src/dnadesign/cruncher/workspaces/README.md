## Workspaces

Put cruncher workspaces in this folder. Each workspace should contain
`configs/config.yaml`, `configs/runbook.yaml`, and a sibling `runbook.md`, plus
any inputs you want to keep alongside it.

Runbook coupling contract:

- `runbook.md` is the human-facing guide.
- `configs/runbook.yaml` is the machine-facing source of truth.
- The `### Step-by-Step Commands` block in `runbook.md` must mirror
  `configs/runbook.yaml` command-for-command in the same order.
- Optional verification or follow-up commands belong in a separate optional
  section so the one-line run command never drifts from the standard sequence.

For optimization workspaces (non-portfolio), keep standard study specs:
`configs/studies/length_vs_score.study.yaml` and
`configs/studies/diversity_vs_score.study.yaml`.

Current layout:

- didactic demos: `demo_pairwise/`, `demo_multitf/`
- pairwise optimization slices:
  `pairwise_cpxr_baer/`,
  `pairwise_cpxr_lexa/`,
  `pairwise_laci_arac/`,
  `pairwise_baer_lexa/`,
  `pairwise_cpxr_soxr/`,
  `pairwise_baer_soxr/`,
  `pairwise_soxr_soxs/`
- multitf optimization slices:
  `multitf_cpxr_baer_lexa/`,
  `multitf_baer_lexa_soxr/`,
  `multitf_baer_lexa_soxr_soxs/`
- operational project workspace: `project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs/`
- portfolio aggregation workspace: `portfolios/`

Example:

```
src/dnadesign/cruncher/workspaces/
  demo_pairwise/
    configs/
      config.yaml
      runbook.yaml             # machine runbook (typed cruncher steps)
      studies/                 # optional study specs
        length_vs_score.study.yaml
        diversity_vs_score.study.yaml
    runbook.md
    inputs/                   # demo inputs
    .cruncher/                # local cache + lockfiles (generated)
    outputs/                  # run outputs (parse/sample/analyze/report)
  portfolios/
    configs/
      runbook.yaml
      master_all_workspaces.portfolio.yaml
    runbook.md
    outputs/                  # portfolio run outputs
```

For the nine pairwise/multitf optimization slices above, keep one shared
`sample`/optimizer hyperparameter profile and vary only:
`workspace.regulator_sets`, discovery/catalog source IDs, and analysis pairwise
projection.

Standard lifecycle for any workspace:

```bash
# Pin config path for repeated CLI calls.
CONFIG="$PWD/configs/config.yaml"
# Fetch TF binding sites from the configured source.
cruncher fetch sites ... -c "$CONFIG"
# Run motif discovery over fetched site evidence.
cruncher discover motifs ... -c "$CONFIG"
# Freeze motif/source provenance for deterministic downstream steps.
cruncher lock -c "$CONFIG"
# Parse inputs into normalized Cruncher artifacts.
cruncher parse --force-overwrite -c "$CONFIG"
# Generate candidate sequences from parsed motif artifacts.
cruncher sample --force-overwrite -c "$CONFIG"
# Compute analysis summaries for generated sequence sets.
# Optional video output: set analysis.trajectory_video.enabled=true in configs/config.yaml.
cruncher analyze --summary -c "$CONFIG"
# Export latest elite sequences for downstream use.
cruncher export sequences --latest -c "$CONFIG"
```

Standard machine runbook execution:

```bash
# Execute the Cruncher machine runbook for this workspace.
cruncher workspaces run --runbook configs/runbook.yaml
```

Standard per-workspace study execution:

```bash
# Run the configured Cruncher parameter-sweep study.
cruncher study run --spec configs/studies/length_vs_score.study.yaml --force-overwrite
# Run the configured Cruncher parameter-sweep study.
cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite
```

Study compaction lifecycle (prune transient trial artifacts after summary plots/tables are emitted):

```bash
# Compact completed study outputs to reduce disk usage.
cruncher study compact --run outputs/studies/<study_name>/<study_id>
# Compact completed study outputs to reduce disk usage.
cruncher study compact --run outputs/studies/<study_name>/<study_id> --confirm
```

`study compact` preserves trial `elites.parquet` and study-level aggregate plots/tables while removing large trial-side
intermediates (`sequences.parquet`, random baseline tables, optimizer move stats sidecar, trace, and stale `*.tmp` files).

If you change `catalog.source_preference` or discovery `--source-id`, re-run `cruncher lock -c "$CONFIG"` before parse.

Workspace hygiene (dry-run by default):

```bash
# Reset workspace outputs before a fresh Cruncher run.
cruncher workspaces reset --root src/dnadesign/cruncher/workspaces --all-workspaces
# Reset workspace outputs before a fresh Cruncher run.
cruncher workspaces reset --root src/dnadesign/cruncher/workspaces
# Reset workspace outputs before a fresh Cruncher run.
cruncher workspaces reset --root src/dnadesign/cruncher/workspaces --confirm
# Reset workspace outputs before a fresh Cruncher run.
cruncher workspaces reset --root src/dnadesign/cruncher/workspaces --all-workspaces --confirm
```

Tip: `cd` into a workspace and run cruncher commands without passing `--config`; Cruncher resolves `configs/config.yaml` automatically. You can also run from anywhere with `--workspace <name>` or inspect what is available via `cruncher workspaces list`.

Packaged workspace configs resolve `discover.tool_path` relative to their `configs/config.yaml`, so keep packaged workspaces under the repository layout (`src/dnadesign/cruncher/workspaces/...`). If you copy a workspace elsewhere, update `discover.tool_path` explicitly before running `discover motifs`.

Portfolio aggregation lifecycle (from a portfolio workspace):

```bash
# Set `SPEC` for later commands in this block.
SPEC="$PWD/configs/master_all_workspaces.portfolio.yaml"
# Run portfolio aggregation across selected workspace studies.
cruncher portfolio run --spec "$SPEC" --prepare-ready skip
# Show aggregated portfolio outputs for this run id.
cruncher portfolio show --run outputs/master_all_workspaces/<portfolio_id>
```

The portfolio template spec uses `portfolio.schema_version: 3` with
`execution.mode: prepare_then_aggregate`, so `cruncher portfolio run` can
prepare source workspaces by executing the declared source runbook steps before
aggregation. The template also declares `studies.ensure_specs` and
`studies.sequence_length_table` so aggregation emits
`tables/table__handoff_sequence_length.*` keyed on `sequence_length`.

Portfolio source `run_dir` guidance:

- single regulator set workspace: `run_dir: outputs`
- multi-set workspace: `run_dir: outputs/set<index>_<tf-slug>`

Portfolio source precondition per included workspace run:

```bash
# Compute analysis summaries for generated sequence sets.
# Optional video output: set analysis.trajectory_video.enabled=true in the source workspace config.
cruncher analyze --summary -c <source_workspace>/configs/config.yaml
# Export latest elite sequences for downstream use.
cruncher export sequences --run <source_run_dir> -c <source_workspace>/configs/config.yaml
```
