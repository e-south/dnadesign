# Workspace Snapshot Contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-15

`latentdna workspace snapshot --workspace <id|path> --json` publishes the
sanctioned study-facing status surface and writes
`outputs/status/workspace_snapshot.json`. Use `--dry-run` when you need the
same `latentdna.workspace_snapshot.v1` payload for inspection without touching
the workspace snapshot file.

Required top-level fields:

- `schema_version`
- `workspace_id`
- `output_root`
- `sources`
- `model_families`
- `canonical_views`
- `candidate_inventory`
- `deliverables`
- `exports`
- `browser`
- `decision_ladder`
- `last_updated_at`

`decision_ladder` publishes the primary decision surfaces only. Gate
deliverables still appear under `deliverables`, but they are not listed as
decision steps.

`candidate_inventory` publishes one machine-readable row per configured
candidate X view. Each row records the candidate set memberships, source,
dataset or path, row basis, model name, feature family, modality, sequence
scope, pooling operation, orientation, coordinate space, role, dimensions when
known, materialization status, and freshness status. Planned or retired
candidate views remain visible in this ledger instead of disappearing from
representation comparisons. The same row contract is also embedded in generated
notebook controls so Marimo startup uses precomputed ledger metadata rather
than reopening candidate matrices just to describe row counts or dimensions.

Promoter-study tooling is allowed to know only:

- the study-owned binding file
- the workspace snapshot schema
- the LatentDNA CLI contract
- exported artifact formats

Promoter-study tooling must not import LatentDNA internals or reconstruct workspace semantics from provider modules.
