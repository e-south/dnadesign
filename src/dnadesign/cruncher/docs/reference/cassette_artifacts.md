## Cassette artifacts

**Owner:** dnadesign-maintainers
**Doc kind:** reference
**Audience:** cassette workflow users and maintainers
**Updated by:** cruncher-maintainers on 2026-03-24
**Applies to:** `uv run cruncher cassette design|solve`
**Last verified:** 2026-03-24
**Primary artifacts:** explicit cassette reports plus solve reports, hit tables, and per-hit bundles

### Contents
- [Run directory](#run-directory)
- [Directory layout](#directory-layout)
- [File semantics](#file-semantics)
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
      render_contract.json        # optional
  export/
    table__candidates.csv
```

### File semantics

- `meta/cassette_manifest.json`: cassette-stage manifest with workspace root, spec path, catalog path, hashes, artifact inventory, spec schema version, and coordinate semantics.
- `meta/cassette_status.json`: lightweight status summary with `completed` or `unsatisfied`, issue count, and schema-mode metadata.
- `provenance/spec_used.yaml`: exact spec snapshot used for the run.
- `provenance/nickase_catalog.yaml`: exact nickase catalog snapshot used for the run.
- `analysis/reports/report.json`: full machine-readable planning report including metadata, issues, intended site instances, intended nick events, and the bounded nicked segment.
- `analysis/reports/report.md`: human-readable planning report.
- `analysis/reports/render_contract.json`: optional dual-view render contract when `output.write_render_contract: true`.
- `export/table__candidates.csv`: one-row satisfied candidate table or header-only CSV for unsatisfied runs.

The render contract carries two views:

- `ssdna_hairpin`
- `linear_duplex`

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
  hits/
    001_<hit_id>/
      resolved_candidate.cassette.yaml
      report.json
      report.md
      manifest.json
      status.json
      render_contract.json    # optional
```

Solve-mode semantics:

- `solve_report.json`: machine-readable solve summary with `solved`, `no_hits`, `invalid_spec`, or `invalid_catalog`.
- `table__hits.csv`: ranked hit table with score, nick boundaries, bounded segment length, and GC metrics.
- `specs/resolved_catalog.yaml`: merged preset-plus-overlay catalog snapshot used for the solve when catalog loading succeeds.
- `hits/<rank>_<hit_id>/resolved_candidate.cassette.yaml`: explicit spec that round-trips through the normal cassette planner.
- per-hit `report.json` and `report.md`: explicit satisfied report for the materialized candidate.
- per-hit `render_contract.json`: only written when `output.write_render_contract: true`.
- `invalid_spec` and `invalid_catalog` preflight failures still write a top-level solve bundle when the workspace and solve output root can be derived, but they do not materialize hit bundles or a resolved catalog snapshot.

### Status behavior

- satisfied specs write `status: completed`
- unsatisfied specs write `status: unsatisfied`
- unsatisfied runs still preserve manifest, status, provenance snapshots, and reports
- solve status artifacts preserve `warning_count`, `warnings`, and `search_truncated` so budget-capped searches are machine-visible without reopening the full report
- cassette runs do not register in workspace `run_index.json`
- cassette runs do not write legacy sample artifacts such as `meta/run_manifest.json`, `optimize/`, or `plots/`
