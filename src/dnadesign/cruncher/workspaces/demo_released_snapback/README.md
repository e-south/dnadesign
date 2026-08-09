# Released snapback demo

**Owner:** dnadesign-maintainers
**Kind:** runnable public example
**Start here:** [`runbook.md`](runbook.md)
**Last verified:** 2026-08-08

This workspace demonstrates a released-product snapback search with packaged
nickase and Type IIS release-enzyme catalogs. It uses one explicit `0/3/3`
geometry so the search, ranking, materialization, and rendering outputs are easy
to inspect.

## Run

```bash
uv run cruncher workspaces run \
  --workspace demo_released_snapback \
  --runbook configs/runbook.yaml
```

The runbook searches the allowed enzyme placements, pins the release enzyme to
`BspQI`, excludes `FREQUENT_CUTTER` nickases, and writes the selected bundles to
`outputs/released_solve/`.

## Review

- `outputs/released_solve/analysis/materialized_hits/` contains ranked hit bundles.
- Each emitted triptych shows the precursor, released fragments, and foldback.
- [`configs/snapback/invalid_origin.released.snapback.yaml`](configs/snapback/invalid_origin.released.snapback.yaml)
  is an intentional negative fixture. `released-design` must reject it before
  publishing a bundle.

The example is not a study or an enzyme recommendation. Change the target
geometry, catalogs, and policy flags explicitly for another use case.
