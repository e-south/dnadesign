# Alignment Contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-11

`alignment_set` is a first-class artifact. Cross-view work that depends on row correspondence must reuse a compiled alignment rather than ad hoc merge logic.

Current supported shape:

- explicit `left` and `right`
- `on: record_key`, `on: subject_key`, or explicit key columns
- `support: intersection`
- explicit aggregation modes on each side

Persisted alignment artifacts include:

- `rows.parquet`
- `mapping.parquet`
- `manifest.json`

Freshness notes:

- alignment manifests now record concrete input paths as well as digests so deliverable status can report stale upstream inputs instead of falling back to `attention` for unknown provenance.

See also:

- [view-contract.md](view-contract.md)
- [artifact-manifests.md](artifact-manifests.md)
