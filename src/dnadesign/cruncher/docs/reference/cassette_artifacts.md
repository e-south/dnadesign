## Cassette artifacts

**Owner:** dnadesign-maintainers
**Doc kind:** reference
**Audience:** cassette workflow users and maintainers
**Updated by:** cruncher-maintainers on 2026-03-24
**Applies to:** `uv run cruncher cassette design`
**Last verified:** 2026-03-24
**Primary artifacts:** cassette manifest, status, reports, provenance snapshots, candidate table

### Contents
- [Run directory](#run-directory)
- [Directory layout](#directory-layout)
- [File semantics](#file-semantics)
- [Status behavior](#status-behavior)

### Run directory

Cassette runs are written under:

```text
<workspace>/outputs/cassettes/<spec.name>/<design_id>/
```

`design_id` is deterministic from the frozen spec bytes plus the frozen catalog bytes. Re-running the same spec and catalog
lands in the same directory unless you change inputs or pass `--force-overwrite` to replace an existing directory.

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

- `meta/cassette_manifest.json`: cassette-stage manifest with workspace root, spec path, catalog path, hashes, and artifact inventory.
- `meta/cassette_status.json`: lightweight status summary with `completed` or `unsatisfied`, issue count, and timestamp.
- `provenance/spec_used.yaml`: exact spec snapshot used for the run.
- `provenance/nickase_catalog.yaml`: exact nickase catalog snapshot used for the run.
- `analysis/reports/report.json`: full machine-readable planning report.
- `analysis/reports/report.md`: human-readable planning report.
- `analysis/reports/render_contract.json`: optional dual-view render contract when `output.write_render_contract: true`.
- `export/table__candidates.csv`: one-row satisfied candidate table or header-only CSV for unsatisfied runs.

The render contract carries two views:

- `ssdna_hairpin`
- `linear_duplex`

### Status behavior

- satisfied specs write `status: completed`
- unsatisfied specs write `status: unsatisfied`
- unsatisfied runs still preserve manifest, status, provenance snapshots, and reports
- cassette runs do not register in workspace `run_index.json`
- cassette runs do not write legacy sample artifacts such as `meta/run_manifest.json`, `optimize/`, or `plots/`
