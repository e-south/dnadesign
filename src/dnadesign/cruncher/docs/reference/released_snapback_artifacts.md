## Released-product snapback artifacts

**Owner:** dnadesign-maintainers
**Doc kind:** reference
**Audience:** snapback workflow users and maintainers
**Last updated by:** cruncher-maintainers on 2026-04-21
**Applies to:** `uv run cruncher snapback released-design|released-show`
**Last verified:** 2026-04-21
**Primary artifacts:** released-product reports, projection payloads, pre-event site records, released-design summary tables

### Contents
- [Run root](#run-root)
- [Bundle layout](#bundle-layout)
- [File semantics](#file-semantics)
- [Status and drift behavior](#status-and-drift-behavior)

### Run root

Released-product snapback uses one stable workspace-relative output root:

```text
<workspace>/outputs/released_design/
```

The lane computes a `released_design_id` for provenance and integrity, but the v1 run directory stays stable so the checked-in runbook can reuse one explicit root.

### Bundle layout

`released-design` writes:

```text
<workspace>/outputs/released_design/
  meta/
    released_snapback_manifest.json
    released_snapback_status.json
  provenance/
    spec.snapshot.yaml
    nickase_catalog.yaml
    release_catalog.yaml
  analysis/
    report.json
    released_product_projection.json
    pre_nick_site.json
    release_site.json
  export/
    released_design_summary.csv
```

### File semantics

- `meta/released_snapback_manifest.json`: bundle manifest with workspace root, spec path, contract name, status, and artifact inventory
- `meta/released_snapback_status.json`: lightweight released-product status record
- `provenance/spec.snapshot.yaml`: exact released-product spec snapshot used for the run
- `provenance/nickase_catalog.yaml`: resolved nickase catalog snapshot
- `provenance/release_catalog.yaml`: resolved release-enzyme catalog snapshot
- `analysis/report.json`: full machine-readable released-product report
- `analysis/released_product_projection.json`: precursor-to-retained-product projection payload
- `analysis/pre_nick_site.json`: resolved pre-nick recognition site plus nick event
- `analysis/release_site.json`: resolved release recognition site plus ds-cut event
- `export/released_design_summary.csv`: one-row released-product summary when a truthful candidate exists, or header-only CSV otherwise

The v1 lane does not publish a large render surface. The projection and site JSON files are the operator-facing analysis contract.

### Status and drift behavior

Released-product explicit bundles can report:

- `satisfied`
- `unsatisfied`
- `invalid_catalog`
- `invalid_precursor`
- `no_release_path`
- `post_release_projection_failed`

`released-show` is strict on purpose. It fails fast when it finds:

- manifest and status contract drift
- missing required provenance snapshots or analysis payloads
- report and projection inconsistency
- run-dir, workflow, or stage drift

`released-show` is an integrity check, not a best-effort artifact browser.
