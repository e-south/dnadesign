## Snapback artifacts

**Owner:** dnadesign-maintainers
**Doc kind:** reference
**Audience:** snapback workflow users and maintainers
**Last updated by:** cruncher-maintainers on 2026-04-21
**Applies to:** `uv run cruncher snapback design|solve|show`
**Last verified:** 2026-04-21
**Primary artifacts:** explicit reports, solve reports, QA triptych contracts, frontier and hit tables, materialized explicit hit bundles

### Contents
- [Run roots](#run-roots)
- [Explicit design layout](#explicit-design-layout)
- [Solve layout](#solve-layout)
- [Materialized hit layout](#materialized-hit-layout)
- [QA and render handoff](#qa-and-render-handoff)
- [Status and drift behavior](#status-and-drift-behavior)

### Run roots

Snapback uses stable workspace-relative output roots:

```text
<workspace>/outputs/design/
<workspace>/outputs/solve/
```

The live lane does not create per-run subdirectories from `design_id` or `solve_id`. Those identifiers still exist in reports and manifests, but the directory roots stay stable for the workspace demo and runbook flow.

### Explicit design layout

Explicit `snapback design` writes:

```text
<workspace>/outputs/design/
  meta/
    snapback_manifest.json
    snapback_status.json
  provenance/
    spec_used.yaml
    nickase_catalog.yaml
  analysis/
    reports/
      report.json
      report.md
    views/
      pre_nick_duplex.v1.json                    # optional when a truthful candidate exists
      post_nick_exposed.v1.json                  # optional when a truthful candidate exists
      post_nick_foldback.v1.json                 # optional when a truthful candidate exists
      pre_nick_duplex.snapback_visual.v1.json    # optional
      post_nick_exposed.snapback_visual.v1.json  # optional
      post_nick_foldback.snapback_visual.v1.json # optional
      snapback_triptych.snapback_visual.v1.jsonl # optional
      views_manifest.v1.json                     # optional
  export/
    table__candidates.csv
  baserender_jobs/
    snapback_triptych.job.yaml                   # optional
  plots/
    snapback_triptych.<png|svg|pdf>              # optional, operator-rendered
```

File semantics:

- `meta/snapback_manifest.json`: explicit bundle manifest with workspace root, spec path, contract name, artifact inventory, and run status
- `meta/snapback_status.json`: lightweight explicit status summary
- `provenance/spec_used.yaml`: exact explicit spec snapshot used for the bundle
- `provenance/nickase_catalog.yaml`: resolved nickase catalog snapshot
- `analysis/reports/report.json`: full machine-readable explicit report
- `analysis/reports/report.md`: human-readable explicit report
- `export/table__candidates.csv`: one-row candidate table for a satisfied design, or header-only CSV when no truthful candidate exists
- `analysis/views/*.v1.json`: producer-owned QA views
- `analysis/views/*.snapback_visual.v1.json`: shared renderer-facing visual contracts
- `analysis/views/snapback_triptych.snapback_visual.v1.jsonl`: ordered three-state contract batch
- `analysis/views/views_manifest.v1.json`: grouped QA/visual inventory plus recommended jobs
- `baserender_jobs/snapback_triptych.job.yaml`: optional downstream BaseRender job; Cruncher publishes it but does not run it

### Solve layout

`snapback solve` writes:

```text
<workspace>/outputs/solve/
  meta/
    solve_manifest.json
    solve_status.json
  provenance/
    input_solve_spec.yaml
    resolved_catalog.yaml
  analysis/
    reports/
      solve_report.json
      solve_report.md
    materialized_hits/
      hit_01/
      hit_02/
      ...
  export/
    table__hits.csv
    table__frontier.csv
```

File semantics:

- `meta/solve_manifest.json`: solve bundle manifest with artifact inventory and active solve contract
- `meta/solve_status.json`: lightweight solve status summary with hit count
- `provenance/input_solve_spec.yaml`: exact solve spec snapshot
- `provenance/resolved_catalog.yaml`: merged nickase catalog snapshot used for the solve run
- `analysis/reports/solve_report.json`: full machine-readable solve report including resolved search space, search counts, frontier rows, hits, warnings, and materialized-hit paths
- `analysis/reports/solve_report.md`: human-readable solve report
- `export/table__hits.csv`: ranked accepted hits
- `export/table__frontier.csv`: compact search-frontier summary
- `analysis/materialized_hits/`: top-ranked explicit hit bundles under stable `hit_<rank>` paths
- the top-level solve bundle intentionally does not scaffold `analysis/views/`, `baserender_jobs/`, or `plots/`; those belong only to materialized explicit hit bundles

### Materialized hit layout

Each materialized solve hit is a normal explicit snapback bundle:

```text
<workspace>/outputs/solve/analysis/materialized_hits/hit_<rank>/
  meta/
    snapback_manifest.json
    snapback_status.json
  provenance/
    spec_used.yaml
    nickase_catalog.yaml
  analysis/
    reports/
      report.json
      report.md
    views/
      ...
  export/
    table__candidates.csv
  baserender_jobs/
    snapback_triptych.job.yaml                   # optional
```

Materialized hits use the explicit `single_nick_snapback_v2` contract, even when they were selected by a `single_nick_snapback_solve_v3` solve bundle.

### QA and render handoff

The snapback publication surface is local and file-based:

1. Cruncher writes three producer-owned QA views.
2. Cruncher writes three shared `snapback_visual_v1` contracts and one JSONL triptych.
3. Cruncher optionally writes one sibling BaseRender job.
4. An operator can run that job later to populate `plots/`.

Snapback does not invoke BaseRender directly, does not reuse cassette or YIU view contracts, and does not append to workspace `run_index.json`.

### Status and drift behavior

Explicit bundles can be `satisfied`, `unsatisfied`, or `invalid_catalog`.

Solve bundles can be `satisfied`, `no_hits`, or `search_truncated` from the live search path.

`snapback show` is strict on purpose. It fails fast when it finds:

- manifest/status workflow, contract, stage, or run-dir drift
- missing required reports or provenance snapshots
- visual-contract drift against the explicit candidate payload
- triptych ordering drift
- materialized-hit count, rank-path, or candidate-identity drift

The show path is an integrity check, not a best-effort browser.
