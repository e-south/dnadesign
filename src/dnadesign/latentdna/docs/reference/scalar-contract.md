# Scalar Contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-11

Scalar tables are first-class artifacts keyed by rows or aligned support.

Current scalar derivations:

- `vector_norm`
- `column_expression`
- `select_columns`
- `rename_columns`
- `join_tables`

Operational notes:

- `vector_norm` depends on an existing persisted view artifact.
- `column_expression` is intentionally narrow and safe.
- `select_columns` and `rename_columns` make export-ready scalar tables without reopening notebook logic.
- `join_tables` performs an explicit inner join over two or more persisted scalar/distance tables on named key columns with deterministic row ordering inherited from the first source.

See also:

- [control-distances.md](../workflows/control-distances.md)
- [export-contract.md](export-contract.md)
