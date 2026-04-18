# Cross-View Agreement

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-17

This workflow compares view structure without mixing incompatible raw coordinate spaces.

Primary checked-in promoter-study role:

- optional appendix or QC-only diagnostic outside the active reference-first decision ladder

Core artifact path:

1. Materialize the compared views.
2. Build a shared sampled scope or aligned support.
3. Fit reusable `neighbor_set` and `cluster_set` artifacts per view.
4. Compare those artifacts with `agreement compare`.
5. Render an `agreement_summary` plot.

Key invariants:

- Agreement operates over neighbors, clusters, and landmark neighborhoods, not raw cross-model subtraction.
- Shared support must be explicit.
- The resulting `agreement_set` is a persisted artifact, not notebook-only logic.
- Agreement summaries are not primary representation-comparison evidence for the active promoter study.

See also:

- [context-shift.md](context-shift.md)
- [deliverable-contract.md](../reference/deliverable-contract.md)
- [performance-budgets.md](../reference/performance-budgets.md)
