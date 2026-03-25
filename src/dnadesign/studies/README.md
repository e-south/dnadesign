![Studies banner](assets/studies-banner.svg)

Study-family packages keep domain-specific study semantics outside `ops` core.
They read checked-in study records from [Study records](../../../docs/studies/README.md),
assemble family-owned snapshot or preflight logic, and expose provider-owned
registry fragments that `ops progress` can discover lazily.

Use this package when:
- you are adding or changing family-specific study snapshot or preflight logic
- you need to register a new study-owned status surface without editing OPS core

Do not use this package when:
- the change belongs to neutral OPS control-plane or observation-shell code
- the change only touches checked-in study records under `docs/studies/`

Current families:
- `promoter`: promoter-study snapshot and preflight adapters

See also:
- [Ops README](../ops/README.md)
- [Study records](../../../docs/studies/README.md)
