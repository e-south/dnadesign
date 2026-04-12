# Landmark Neighborhoods

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-11

This workflow covers the control-neighborhood enrichment slice built from persisted view and neighbor artifacts.

Primary checked-in promoter-study deliverable:

- `control_neighborhood_enrichment`

Core artifact path:

1. Materialize the anchor-space view.
2. Fit a reusable `neighbor_set` on that view or on an explicit sampled scope.
3. Score enrichment against a declared cohort and one or more declared landmarks.
4. Render the resulting `enrichment_set` as a `heatmap` plot.

Key invariants:

- Landmarks must be declared explicitly in workspace config.
- Enrichment runs from persisted `neighbor_set` artifacts only.
- Plot rendering stays read-only and fails if enrichment artifacts are missing.

See also:

- [control-distances.md](control-distances.md)
- [deliverable-contract.md](../reference/deliverable-contract.md)
- [alignment-contract.md](../reference/alignment-contract.md)
