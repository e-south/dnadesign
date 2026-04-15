# Export Contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-11

`export matrix` and `export table` build deterministic handoff bundles with:

- `rows.parquet`
- `features.parquet`
- `matrix.npy` or `table.parquet`
- `manifest.json`

Current block kinds:

- `reduced_view`
- `table_columns`

Key constraints:

- row basis is explicit and may be a view, reduced view, sample, alignment, scalar table, or distance table artifact.
- block order is stable and encoded in `features.parquet`.
- alignment-backed block projection is explicit.
- feature names use deterministic prefixes such as `z20_60_pc_001`.
- feature names must be unique across the full bundle; ambiguous ledgers fail fast.

See also:

- [deliverable-contract.md](deliverable-contract.md)
- [performance-budgets.md](performance-budgets.md)
