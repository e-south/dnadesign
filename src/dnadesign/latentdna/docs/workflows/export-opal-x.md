# Export To OPAL X Bundles

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-11

This workflow covers deterministic export bundles that hand off explicit `X` matrices to downstream supervised tools.

Primary checked-in promoter-study deliverables:

- `x0_primary_20b`
- `x1_primary_20b`
- `x2_primary_20b`
- `x3_ablation_7b`

Core artifact path:

1. Materialize the required source-backed or derived views.
2. Fit reducers once and persist reduced views.
3. Score any scalar or landmark-distance sidecar tables.
4. Export a deterministic matrix or aligned table bundle with a feature ledger.

Key invariants:

- Export blocks are ordered explicitly.
- Alignment-backed block projection must name the alignment and aggregation mode.
- Metadata stays outside the numeric matrix.
- Feature names are stable and ledger-backed.

See also:

- [promoter-study-latent-atlas.md](promoter-study-latent-atlas.md)
- [export-contract.md](../reference/export-contract.md)
- [performance-budgets.md](../reference/performance-budgets.md)
