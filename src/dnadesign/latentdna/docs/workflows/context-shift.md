# Context Geometry

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-11

This workflow covers paired anchor-versus-full-context geometry metrics computed on explicit matched supports.

Primary checked-in promoter-study deliverable:

- `context_geometry_audit`

Core artifact path:

1. Materialize the paired anchor and full-context views for one representation family and model family.
2. Compile the explicit `alignment_set` on the matched subject key.
3. Compute paired geometry metrics such as `context_self_cosine`, `context_shift_l2`, and margin deltas.
4. Persist the paired sample ids used for geometry-distance correlation and kNN-overlap checks.
5. Render summary tables and plots only from persisted artifacts.

Key invariants:

- Paired comparisons require matching coordinate-space ids and explicit alignment support.
- Geometry preservation metrics operate on persisted paired sample ids, not ad hoc notebook subsets.
- This workflow supports whole-sequence pooled representation shift claims only; it does not support anchor-local mechanistic claims.

See also:

- [cross-view-agreement.md](cross-view-agreement.md)
- [view-contract.md](../reference/view-contract.md)
- [alignment-contract.md](../reference/alignment-contract.md)
