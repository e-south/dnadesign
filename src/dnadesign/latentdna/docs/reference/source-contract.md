# Source Contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-11

`latentdna` currently accepts three source kinds:

- `usr`
- `parquet`
- `matrix_bundle`

`matrix_bundle` is the file-backed source form for a precomputed matrix plus its row ledger. The current contract expects:

- `rows.parquet`
- `matrix.npy` or `matrix.npz`
- declared `record_key` and `subject_key` columns in `rows.parquet`

Important constraints:

- `vector.kind: bundle_matrix` is legal only for `matrix_bundle` sources.
- `vector.kind: column` is illegal for `matrix_bundle` sources.
- materialization validates row-count parity between the matrix payload and `rows.parquet`.
- source-backed view materialization canonicalizes the matrix into workspace-owned outputs rather than aliasing the external bundle in place.

See also:

- [workspace-schema.md](workspace-schema.md)
- [view-contract.md](view-contract.md)
