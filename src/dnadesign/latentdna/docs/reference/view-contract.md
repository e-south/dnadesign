# View Contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-11

Source-backed view kinds:

- `vector.kind: column`
- `vector.kind: bundle_matrix`

Current derived view kinds:

- `vector_difference`
- `normalize`
- `aggregate_by_key`
- `apply_reducer`
- `concatenate`

Key legality rules:

- `vector_difference` requires identical coordinate-space ids, matching dimensionality, and explicit alignment support.
- `normalize` preserves the source coordinate-space id only when the declared method keeps the view in the same semantic space.
- `apply_reducer` consumes a persisted reducer artifact and creates a new reduced coordinate space.
- `concatenate` is intended for already-compatible reduced or normalized blocks, not silent raw cross-model mixing.

See also:

- [alignment-contract.md](alignment-contract.md)
- [export-contract.md](export-contract.md)
