## Portfolio aggregation

**Last updated by:** cruncher-maintainers on 2026-02-23


### Contents
- [Why this exists](#why-this-exists)
- [Required source readiness](#required-source-readiness)
- [Spec layout](#spec-layout)
- [Canonical command sequence](#canonical-command-sequence)
- [Outputs](#outputs)

### Why this exists

`study` aggregates trial sweeps inside one workspace. `portfolio` aggregates selected completed runs across multiple workspaces into one handoff package.

Use `portfolio` when you need one export-ready table for experimental follow-up across pairwise/multitf slices.

### Required source readiness

`portfolio.execution.mode` determines readiness requirements:

- `aggregate_only`: each source run must already have `analysis/reports/summary.json` and
  `export/export_manifest.json` with a valid `files.elites` table path.
- `prepare_then_aggregate`: each source must provide `prepare.runbook` + `prepare.step_ids`; those steps must produce
  the required source artifacts before aggregation.

Portfolio is fail-fast: preparation errors, missing source artifacts, or contract violations abort the run.

To inspect available study specs before wiring a portfolio, run:

```bash
cruncher study list --workspace <workspace_name_or_path>
```

### Spec layout

Store portfolio specs inside the portfolio workspace:

```text
<portfolio-workspace>/configs/<name>.portfolio.yaml
```

Minimal shape:

```yaml
portfolio:
  schema_version: 3
  name: master_all_workspaces
  execution:
    mode: prepare_then_aggregate
    max_parallel_sources: 4
  studies:
    ensure_specs:
      - configs/studies/length_vs_score.study.yaml
      - configs/studies/diversity_vs_score.study.yaml
    sequence_length_table:
      enabled: true
      study_spec: configs/studies/length_vs_score.study.yaml
      top_n_lengths: 6
  artifacts:
    table_format: parquet
    write_csv: false
  plots:
    elite_showcase:
      enabled: true
      top_n_per_source: 3
      ncols: 5
      plot_format: pdf
      dpi: 250
      source_selectors:
        pairwise_cpxr_baer:
          elite_ranks: [1, 3]
        pairwise_cpxr_lexa:
          elite_ids: [pairwise_cpxr_lexa_elite_1, pairwise_cpxr_lexa_elite_2]
  sources:
    - id: pairwise_cpxr_baer
      workspace: ../pairwise_cpxr_baer
      run_dir: outputs
      study_spec: configs/studies/diversity_vs_score.study.yaml
      prepare:
        runbook: configs/runbook.yaml
        step_ids: [fetch_sites_regulondb, discover_motifs, render_logos, lock_targets, parse_run, sample_run, analyze_summary, export_sequences_latest]
    - id: pairwise_cpxr_lexa
      workspace: ../pairwise_cpxr_lexa
      run_dir: outputs
      study_spec: configs/studies/diversity_vs_score.study.yaml
      prepare:
        runbook: configs/runbook.yaml
        step_ids: [fetch_sites_regulondb, discover_motifs, render_logos, lock_targets, parse_run, sample_run, analyze_summary, export_sequences_latest]
```

For single-regulator-set workspaces, sample outputs live at `outputs/`.
For multi-set workspaces, use the specific set directory (for example `outputs/set2_lexA-cpxR`).

Strict contracts:

- unknown keys fail (`extra=forbid`)
- source IDs must be unique and slug-safe
- every source must set explicit `workspace` and `run_dir`
- in `prepare_then_aggregate` mode, every source must set `prepare.runbook` and non-empty `prepare.step_ids`
- `execution.max_parallel_sources` defaults to `4` and must be `>= 1`
- `studies.ensure_specs` entries are validated for every source workspace and auto-run when missing/incomplete
- `studies.sequence_length_table` is a global portfolio option; it writes one extra table keyed by `sequence_length`
- `plots.elite_showcase` is enabled by default and writes a cross-workspace elite showcase plot using all selected source elites
- set `plots.elite_showcase.top_n_per_source` to cap how many elites per source are rendered
- cross-workspace showcase score tokens are sourced from `best_score_norm` and must remain normalized in `[0,1]` per TF
- `plots.elite_showcase.source_selectors` supports per-source multi-elite selection; each selector must set exactly one of `elite_ids` or `elite_ranks`
- `study_spec` is optional; when set, portfolio writes study summary rows from that deterministic study run
- source selection uses source run manifest `top_k` and the export manifest `files.elites` table
- source run manifest `top_k` must be `>= 1` and must match export elites row count
- source run manifest stage must be `sample`
- `run_dir` must remain inside its declared workspace path
- in `aggregate_only`, `run_dir` must already exist; in `prepare_then_aggregate`, it is validated after prepare steps run
- `prepare.runbook` must exist and remain inside its declared source workspace path

### Canonical command sequence

```bash
set -euo pipefail

# 1) Run portfolio aggregation from the portfolio workspace.
#    In schema v3 prepare_then_aggregate mode, source runbooks are executed first.
cd ../portfolios
uv run cruncher portfolio run --spec configs/master_all_workspaces.portfolio.yaml --prepare-ready skip
uv run cruncher portfolio show --run outputs/portfolios/master_all_workspaces/<portfolio_id>

# 2) If some sources are already ready in prepare_then_aggregate mode,
#    choose explicit behavior:
#    - skip: prepare only missing/unready sources
#    - rerun: run prepare steps for every source
uv run cruncher portfolio run --spec configs/master_all_workspaces.portfolio.yaml --prepare-ready skip
uv run cruncher portfolio run --spec configs/master_all_workspaces.portfolio.yaml --prepare-ready rerun
```

`--spec` must point to the `.portfolio.yaml` file path. Passing `configs/` (directory only) fails fast.

`aggregate_only` now performs a source preflight and reports all missing/invalid source artifacts in one error with source IDs and workspace/run paths.
When missing artifacts indicate the source run itself is not bootstrapped (for example missing run manifest/elites), the error includes a full-runbook nudge (no `--step`) so preparation can build the run from scratch.
Those nudge commands use `--workspace <source_workspace_path>` so they work even when the source workspace is outside discovered workspace-name roots.

### Outputs

Portfolio outputs are deterministic:

```text
<portfolio-workspace>/outputs/portfolios/<portfolio_name>/<portfolio_id>/
```

Canonical handoff table export root:

- `<portfolio-workspace>/outputs/export/portfolios/<portfolio_name>/<portfolio_id>/`

Primary artifacts:

- `outputs/export/portfolios/<portfolio_name>/<portfolio_id>/table__handoff_windows_long.{csv|parquet}`
- `outputs/export/portfolios/<portfolio_name>/<portfolio_id>/table__handoff_elites_summary.{csv|parquet}`
- `outputs/export/portfolios/<portfolio_name>/<portfolio_id>/table__source_summary.{csv|parquet}`
- `outputs/export/portfolios/<portfolio_name>/<portfolio_id>/table__study_summary.{csv|parquet}` (when one or more sources define `study_spec`)
- `outputs/export/portfolios/<portfolio_name>/<portfolio_id>/table__handoff_sequence_length.{csv|parquet}` (when `studies.sequence_length_table.enabled: true`)
- `<portfolio-workspace>/outputs/plots/portfolio__<portfolio_name>__<portfolio_id>__plot__source_tradeoff_score_vs_diversity.pdf` (when diversity metric is available)
- `<portfolio-workspace>/outputs/plots/portfolio__<portfolio_name>__<portfolio_id>__plot__elite_showcase_cross_workspace.<pdf|png>` (when `plots.elite_showcase.enabled: true`)
- `portfolio/portfolio_manifest.json`
- `portfolio/portfolio_status.json`
- `portfolio/logs/prepare__<source_id>.log` (prepare step logs per source, in `prepare_then_aggregate` mode)
- `manifests/manifest.json` + `manifests/table_manifest.json` + `manifests/plot_manifest.json`

Default portfolio table contract is single-format parquet (`artifacts.table_format: parquet`, `artifacts.write_csv: false`).
Enable `write_csv: true` only when a downstream consumer explicitly requires CSV mirrors.

`table__handoff_windows_long` is the tidy handoff table (one row per elite/TF window with sequence, regulator, position, strand, score, and provenance).
`table__handoff_elites_summary` is the minimal elite-level handoff table (one row per elite with deterministic hash ID and sequence-level summaries).
`table__source_summary` provides per-workspace context (selected count, score summaries, and diversity summary).
`table__handoff_sequence_length` provides the first `top_n_lengths` shortest `sequence_length` rows per source from
the configured length study, keeping the best median score row per length.

Both handoff tables include deterministic hash IDs:

- `elite_hash_id`: stable per source/elite sequence row
- `window_hash_id`: stable per source/elite/TF-window row (`table__handoff_windows_long` only)
