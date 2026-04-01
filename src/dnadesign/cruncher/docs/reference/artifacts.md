## Cruncher Artifacts Reference

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-28


**Last updated by:** cruncher-maintainers on 2026-02-28

### Contents
- [Overview](#overview)
- [Run directory layout](#run-directory-layout)
- [Required artifacts by command](#required-artifacts-by-command)
- [Operational entrypoints](#operational-entrypoints)
- [Study and portfolio artifacts](#study-and-portfolio-artifacts)

### Overview
This is the standard artifact contract for Cruncher outputs. Use this page as the source of truth for what each command must produce and what downstream commands consume.

### Run directory layout
Each run directory uses:

```text
<run_dir>/
  meta/
  provenance/
  optimize/
    tables/
    state/
  analysis/
    reports/
    tables/
    manifests/
  plots/
  export/
    sequences/
```

### Required artifacts by command
#### `cruncher sample`
- `meta/run_manifest.json`
- `meta/run_status.json`
- `meta/config_used.yaml`
- `provenance/lockfile.json`
- `optimize/tables/sequences.parquet`
- `optimize/tables/elites.parquet`
- `optimize/tables/elites_hits.parquet` for representative-hit objectives
- `optimize/tables/elites_objective_scores.parquet` for occurrence-aware objectives
- `optimize/tables/elites_occurrences.parquet` for occurrence-aware objectives
- `optimize/state/elites.yaml`
- `optimize/state/metrics.jsonl`

Random baseline sidecars follow the same contract split:
- `optimize/tables/random_baseline.parquet`
- `optimize/tables/random_baseline_hits.parquet` for representative-hit objectives
- `optimize/tables/random_baseline_objective_scores.parquet` for occurrence-aware objectives
- `optimize/tables/random_baseline_occurrences.parquet` for occurrence-aware objectives

#### `cruncher analyze`
Consumes:
- `optimize/tables/sequences.parquet`
- `optimize/tables/elites.parquet`
- `optimize/tables/elites_hits.parquet` when `meta/run_manifest.json` reports `representative_hit_contract: true`
- `optimize/tables/elites_occurrences.parquet` when `meta/run_manifest.json` reports `representative_hit_contract: false`
- matching random-baseline hits or occurrences sidecars when random baselines are enabled

Produces:
- `analysis/reports/summary.json`
- `analysis/reports/report.md`
- `analysis/reports/report.json`
- `analysis/manifests/manifest.json`
- `analysis/manifests/plot_manifest.json`
- `analysis/manifests/table_manifest.json`
- `analysis/tables/table__*.parquet`
- `plots/*.pdf` (or configured format)
- `plots/chain_trajectory_video.mp4` (optional, when `analysis.trajectory_video.enabled=true`)

#### `cruncher export sequences`
Consumes:
- latest analyzed sample run artifacts
- requires `representative_hit_contract: true` in the current v1 export path

Produces:
- `export/table__elites.csv`
- `export/export_manifest.json`
- `export/table__consensus_sites.<csv|parquet>`

Occurrence-aware contract notes:
- Multiplicity runs do not silently redefine `elites_hits.parquet`.
- `cruncher analyze` normalizes `elites_occurrences.parquet` into internal placement rows so the static plot suite can still render occurrence-aware panels.
- Use `meta/run_manifest.json` to inspect `artifact_contract_version`, `representative_hit_contract`, `objective_kinds`, and `occurrence_artifacts` before consuming hit-shaped tables.
- When `representative_hit_contract: false`, downstream readers must use the objective/occurrence sidecars or fail fast.

### Operational entrypoints
- Run summary: `analysis/reports/summary.json`
- Human report: `analysis/reports/report.md`
- Plot inventory: `analysis/manifests/plot_manifest.json`
- Elite export table: `export/table__elites.csv`
- Export inventory: `export/export_manifest.json`

### Study and portfolio artifacts
- Study outputs: `outputs/studies/<study_name>/<study_id>/`
- Study aggregate plots: `outputs/plots/study__<study_name>__<study_id>__plot__*.pdf`
- Portfolio outputs: `outputs/<portfolio_name>/<portfolio_id>/`
- Portfolio aggregate plots: `outputs/<portfolio_name>/<portfolio_id>/plots/plot__*.pdf`
