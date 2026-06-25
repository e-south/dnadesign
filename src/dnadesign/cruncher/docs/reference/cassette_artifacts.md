## Cassette artifacts

**Owner:** dnadesign-maintainers
**Doc kind:** reference
**Audience:** cassette workflow users and maintainers
**Last updated by:** cruncher-maintainers on 2026-04-05
**Applies to:** `uv run cruncher cassette design|solve`
**Last verified:** 2026-06-24
**Primary artifacts:** explicit cassette reports plus shared view contracts, solve reports, hit tables, top-hit JSONL batches, baserender jobs, and per-hit bundles

### Contents
- [Run directory](#run-directory)
- [Directory layout](#directory-layout)
- [File semantics](#file-semantics)
- [Render handoff flow](#render-handoff-flow)
- [Status behavior](#status-behavior)

### Run directory

Explicit cassette runs are written under:

```text
<workspace>/outputs/cassettes/<spec.name>/<design_id>/
```

`design_id` is deterministic from the frozen spec bytes plus the frozen catalog bytes. Re-running the same spec and catalog lands in the same directory unless you change inputs or pass `--force-overwrite` to replace an existing directory.

### Directory layout

```text
<run_dir>/
  meta/
    cassette_manifest.json
    cassette_status.json
  provenance/
    spec_used.yaml
    nickase_catalog.yaml
  analysis/
    reports/
      report.json
      report.md
  export/
    table__candidates.csv
  views/
    linear_duplex.v1.json         # optional
    ssdna_hairpin.v1.json         # optional
    views_manifest.v1.json        # optional
  baserender_jobs/
    linear_duplex.job.yaml        # optional
    ssdna_hairpin.job.yaml        # optional
  renders/
    linear_duplex.pdf            # optional, operator-rendered
    ssdna_hairpin.pdf            # optional, operator-rendered
```

### File semantics

- `meta/cassette_manifest.json`: cassette-stage manifest with workspace root, spec path, catalog path, hashes, artifact inventory, spec schema version, and coordinate semantics.
- `meta/cassette_status.json`: lightweight status summary with `completed` or `unsatisfied`, issue count, and schema-mode metadata.
- `provenance/spec_used.yaml`: exact spec snapshot used for the run.
- `provenance/nickase_catalog.yaml`: exact nickase catalog snapshot used for the run.
- `analysis/reports/report.json`: full machine-readable planning report including metadata, issues, intended site instances, intended nick events, and the bounded nicked segment.
- `analysis/reports/report.md`: human-readable planning report.
- `export/table__candidates.csv`: one-row satisfied candidate table or header-only CSV for unsatisfied runs.
- `views/linear_duplex.v1.json`: shared duplex QA contract when `output.emit_visual_contracts: true` and the planner materializes a concrete candidate.
- `views/ssdna_hairpin.v1.json`: shared hairpin-topology QA contract when `output.emit_visual_contracts: true` and the planner materializes a concrete candidate.
- `views/views_manifest.v1.json`: discovery manifest that groups the emitted view files and recommended jobs when view publication succeeds.
- `baserender_jobs/linear_duplex.job.yaml`: optional baserender job for the duplex QA sheet when the matching duplex view exists.
- `baserender_jobs/ssdna_hairpin.job.yaml`: optional baserender job for the hairpin QA figure when the matching hairpin view exists.
- `renders/*.pdf`: optional baserender outputs written in place when an operator runs the emitted jobs. Cruncher does not invoke baserender automatically.

### Solve directory layout

Cassette solve runs are written under:

```text
<workspace>/outputs/cassette_solves/<solve_id>/
```

Solve layout:

```text
<solve_run_dir>/
  solve_report.json
  solve_report.md
  table__hits.csv
  solve_manifest.json
  solve_status.json
  specs/
    input_solve_spec.yaml
    resolved_catalog.yaml
  views/
    top_hits.linear_duplex.v1.jsonl
    top_hits.ssdna_hairpin.v1.jsonl
  baserender_jobs/
    top_hits_duplex.job.yaml
    top_hits_hairpin.job.yaml
  renders/
    top_hits_duplex_qa_sheet.pdf    # optional, operator-rendered
    top_hits_hairpin_qa_sheet.pdf   # optional, operator-rendered
  hits/
    hit_001_<solution_id>/
      explicit/
        resolved_candidate.cassette.yaml
        report.json
        report.md
        manifest.json
        status.json
      views/
        linear_duplex.v1.json
        ssdna_hairpin.v1.json
        views_manifest.v1.json
      baserender_jobs/
        linear_duplex.job.yaml
        ssdna_hairpin.job.yaml
      renders/
        linear_duplex.pdf           # optional, operator-rendered
        ssdna_hairpin.pdf           # optional, operator-rendered
```

Solve-mode semantics:

- `solve_report.json`: machine-readable solve summary with `solved`, `no_hits`, `invalid_spec`, or `invalid_catalog`, plus `selection_summary` for accepted-pool and policy telemetry.
- `solve_manifest.json`: solve-level manifest with source spec hash plus resolved catalog path/hash and emitted artifact inventory.
- `table__hits.csv`: ranked hit table with score, `solution_id`, `explicit_design_id`, `views_manifest_path`, per-hit job paths, nick boundaries, bounded segment length, GC metrics, and selection telemetry.
- `views/top_hits.linear_duplex.v1.jsonl`: selected-hit duplex QA contracts for multi-hit rendering.
- `views/top_hits.ssdna_hairpin.v1.jsonl`: selected-hit hairpin QA contracts for multi-hit rendering.
- `baserender_jobs/top_hits_duplex.job.yaml`: solve-level baserender job for the duplex contact sheet.
- `baserender_jobs/top_hits_hairpin.job.yaml`: solve-level baserender job for the hairpin contact sheet when enabled.
- top-level `renders/*.pdf`: solve-level baserender outputs written in place when you run the emitted jobs with `baserender`.
- `specs/resolved_catalog.yaml`: merged preset-plus-overlay catalog snapshot used for the solve when catalog loading succeeds.
- `hits/hit_<rank>_<solution_id>/explicit/resolved_candidate.cassette.yaml`: explicit spec that round-trips through the normal cassette planner.
- per-hit `explicit/report.json` and `explicit/report.md`: explicit satisfied report for the materialized candidate.
- per-hit `views/*.json`: shared single-hit view contracts.
- per-hit `baserender_jobs/*.job.yaml`: single-hit baserender jobs that render from the published view files in place.
- per-hit `renders/*.pdf`: optional per-hit baserender outputs written next to the corresponding view/job bundle.
- `invalid_spec` and `invalid_catalog` preflight failures still write a top-level solve bundle when the workspace and solve output root can be derived, but they do not materialize hit bundles or a resolved catalog snapshot.

### Render handoff flow

The cassette render path is intentionally local to the owning workspace and solve bundle:

1. Cruncher writes shared view contracts into `views/`.
2. If job emission is enabled, Cruncher writes sibling `baserender_jobs/*.job.yaml` files that point only at those local `views/` files.
3. BaseRender reads the local job file and writes PDFs into sibling `renders/`.

That rule holds for both explicit runs and solve runs:

- explicit runs stay under `<workspace>/outputs/cassettes/<spec.name>/<design_id>/...`
- solve runs stay under `<workspace>/outputs/cassette_solves/<solve_id>/...`
- per-hit renders stay inside `hits/hit_<rank>_<solution_id>/renders/`

There is no separate baserender workspace contract and no cassette registration in workspace `run_index.json`.

### Status behavior

- satisfied specs write `status: completed`
- unsatisfied specs write `status: unsatisfied`
- unsatisfied runs still preserve manifest, status, provenance snapshots, and reports
- solve status artifacts preserve `warning_count`, `warnings`, `warning_codes`, `search_truncated`, `accepted_pool_truncated`, top-hit JSONL/job paths, and lightweight `selection` telemetry so budget-capped or policy-limited searches are machine-visible without reopening the full report
- warning codes include `MAX_SEARCH_NODES_REACHED`, `MAX_ENUMERATED_CANDIDATES_REACHED`, `ACCEPTED_POOL_TRUNCATED`, `SELECTION_RESULTS_POOL_BOUNDED`, `SELECTION_RESULTS_SEARCH_BOUNDED`, and `SELECTION_POLICY_LIMITED_HITS`
- cassette runs do not register in workspace `run_index.json`
- cassette runs do not write legacy sample artifacts such as `meta/run_manifest.json`, `optimize/`, or `plots/`
