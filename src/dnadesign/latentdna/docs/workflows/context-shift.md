# Context Shift

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-11

This workflow covers within-model deltas such as `delta20` and `delta7` that depend on explicit alignment support.

Primary checked-in promoter-study deliverables:

- `context_shift_primary`
- `drag_qc`

Core artifact path:

1. Materialize the paired anchor and context views.
2. Compile the explicit `alignment_set` on `subject_key`.
3. Derive the vector difference as a persisted view.
4. Derive scalar summaries such as `delta20_norm`.
5. Optionally reduce the delta view for downstream export or plotting.

Key invariants:

- `vector_difference` requires matching coordinate-space ids and explicit alignment.
- Aggregation before alignment must be named, not implicit.
- Delta QC plots render from persisted scalar artifacts only.

See also:

- [cross-view-agreement.md](cross-view-agreement.md)
- [view-contract.md](../reference/view-contract.md)
- [alignment-contract.md](../reference/alignment-contract.md)
