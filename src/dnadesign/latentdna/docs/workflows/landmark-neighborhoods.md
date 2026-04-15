# Landmark Neighborhoods

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-14

This workflow owns the landmark-neighborhood analysis lane, but the current
checked-in promoter-study slice is a reduction QA surface rather than a true
neighborhood enrichment artifact.

Primary checked-in promoter-study deliverable:

- `control_pca_explained_variance_curve`

Core artifact path:

1. Materialize the anchor-space view.
2. Fit a reusable PCA reducer on that view.
3. Render the resulting explained-variance scree as a `curve` plot.

Key invariants:

- The checked-in deliverable name must match the reducer-backed scree implementation.
- Plot rendering stays read-only and fails if reducer artifacts are missing.
- Future neighborhood-enrichment work must add an explicit `enrichment_set`
  artifact instead of reusing the scree surface name.

See also:

- [control-distances.md](control-distances.md)
- [deliverable-contract.md](../reference/deliverable-contract.md)
- [alignment-contract.md](../reference/alignment-contract.md)
