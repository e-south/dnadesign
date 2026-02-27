## Cruncher Artifacts Reference

**Last updated by:** cruncher-maintainers on 2026-02-23

### Contents
- [Overview](#overview)
- [Run directory layout](#run-directory-layout)
- [Required artifacts by command](#required-artifacts-by-command)
- [Operational entrypoints](#operational-entrypoints)
- [Study and portfolio artifacts](#study-and-portfolio-artifacts)

### Overview
This is the canonical artifact contract for Cruncher outputs. Use this page as the source of truth for what each command must produce and what downstream commands consume.

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
- `optimize/tables/elites_hits.parquet`
- `optimize/state/elites.yaml`
- `optimize/state/metrics.jsonl`

#### `cruncher analyze`
Consumes:
- `optimize/tables/sequences.parquet`
- `optimize/tables/elites.parquet`
- `optimize/tables/elites_hits.parquet`

Produces:
- `analysis/reports/summary.json`
- `analysis/reports/report.md`
- `analysis/reports/report.json`
- `analysis/manifests/manifest.json`
- `analysis/manifests/plot_manifest.json`
- `analysis/manifests/table_manifest.json`
- `analysis/tables/table__*.parquet`
- `plots/*.pdf` (or configured format)

#### `cruncher export sequences`
Consumes:
- latest analyzed sample run artifacts

Produces:
- `export/table__elites.csv`
- `export/export_manifest.json`
- `export/table__consensus_sites.<csv|parquet>`

### Operational entrypoints
- Run summary: `analysis/reports/summary.json`
- Human report: `analysis/reports/report.md`
- Plot inventory: `analysis/manifests/plot_manifest.json`
- Elite export table: `export/table__elites.csv`
- Export inventory: `export/export_manifest.json`

### Study and portfolio artifacts
- Study outputs: `outputs/studies/<study_name>/<study_id>/`
- Study aggregate plots: `outputs/plots/study__<study_name>__<study_id>__plot__*.pdf`
- Portfolio outputs: `outputs/portfolios/<portfolio_name>/<portfolio_id>/`
- Portfolio aggregate plots: `outputs/portfolios/<portfolio_name>/<portfolio_id>/plots/plot__*.pdf`
