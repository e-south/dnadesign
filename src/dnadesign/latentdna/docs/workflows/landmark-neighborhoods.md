# Landmark Neighborhoods

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-14

This workflow owns the landmark-neighborhood analysis lane, but the current
checked-in promoter-study slice is a reduction QA surface rather than a true
neighborhood enrichment artifact.

Current role in the promoter-study refactor:

- internal reducer diagnostics only

Representative artifact path:

1. Materialize the anchor-space view.
2. Fit a reusable PCA reducer on that view.
3. Record explained-variance metadata for maintainer-side diagnostics.

Key invariants:

- Diagnostic reducer metadata stays internal and is not a named study deliverable.
- Any optional scree rendering stays read-only and fails if reducer artifacts are missing.
- Future neighborhood-enrichment work must add an explicit `enrichment_set`
  artifact instead of reusing the scree surface name.

See also:

- [control-distances.md](control-distances.md)
- [deliverable-contract.md](../reference/deliverable-contract.md)
- [alignment-contract.md](../reference/alignment-contract.md)
