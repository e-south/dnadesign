## Released-product snapback artifacts

**Owner:** dnadesign-maintainers
**Doc kind:** reference
**Audience:** snapback workflow users and maintainers
**Last updated by:** cruncher-maintainers on 2026-04-22
**Applies to:** `uv run cruncher snapback released-design|released-solve|released-show`
**Last verified:** 2026-04-22
**Primary artifacts:** released-product solve reports, per-hit bundles, projection payloads, pre-event site records, released-design summary tables

### Contents
- [Run root](#run-root)
- [Bundle layout](#bundle-layout)
- [File semantics](#file-semantics)
- [Status and drift behavior](#status-and-drift-behavior)

### Run root

Released-product snapback uses two stable workspace-relative output roots:

```text
<workspace>/outputs/released_solve/
<workspace>/outputs/released_design/
```

The v1 run directory stays stable so an operator-authored runbook can reuse one explicit root when a checked-in released-product precursor exists.

### Bundle layout

`released-solve` writes:

```text
<workspace>/outputs/released_solve/
  meta/
    released_solve_manifest.json
    released_solve_status.json
  provenance/
    request.snapshot.yaml
    nickase_catalog.yaml
    release_catalog.yaml
  analysis/
    solve_report.json
    materialized_hits/
      hit_01/
        ...
      hit_02/
        ...
  export/
    table__hits.csv
```

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

- `meta/released_solve_manifest.json`: solve-bundle manifest with workspace root, artifact inventory, catalog-source labels, and the selected hit kind
- `meta/released_solve_status.json`: lightweight released-product solve status record
- `provenance/request.snapshot.yaml`: exact released-solve request and output settings used for the run
- `analysis/solve_report.json`: full machine-readable released-solve report with embedded search evidence and materialized-hit paths
- `analysis/materialized_hits/hit_<rank>/`: released-product hit bundles with `target_search_hit.json`, projection/site payloads, a plot context JSON that records physical top/bottom fragment-row placement plus the origin-anchored active foldback, and an optional rendered triptych
- `export/table__hits.csv`: ranked hit summary table with route-policy columns (`final_geometry_source`, `route_family`, `active_strand`, `retained_partner_strand`, `physical_nicked_strand`), retained-partner fragment length, generic active-product metrics, and materialized bundle/render paths
- `meta/released_snapback_manifest.json`: bundle manifest with workspace root, spec path, contract name, status, artifact inventory, catalog-source labels, and the pinned `final_target`
- `meta/released_snapback_status.json`: lightweight released-product status record
- `provenance/spec.snapshot.yaml`: exact released-product spec snapshot used for the run
- `provenance/nickase_catalog.yaml`: resolved nickase catalog snapshot
- `provenance/release_catalog.yaml`: resolved release-enzyme catalog snapshot
- `analysis/report.json`: full machine-readable released-product report
- `analysis/released_product_projection.json`: precursor-to-post-release projection payload with the retained active strand, surviving partner fragment, and final geometry source called out explicitly
- `analysis/pre_nick_site.json`: resolved pre-nick recognition site plus nick event
- `analysis/release_site.json`: resolved release recognition site plus ds-cut event
- `export/released_design_summary.csv`: one-row released-product summary when a truthful candidate exists, or header-only CSV otherwise; the row carries route-policy, retained-partner fragment, and generic active-product metrics

The solve lane publishes a large per-hit render surface. The explicit
`released-design` bundle stays intentionally small and projection-centric.

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
- manifest artifact-inventory drift
- spec provenance hash drift
- provenance snapshot hash drift
- missing required provenance snapshots or analysis payloads
- report and projection inconsistency
- report and manifest provenance-label drift
- report and manifest final-target drift
- run-dir, workflow, or stage drift

`released-show` is an integrity check, not a best-effort artifact browser.
