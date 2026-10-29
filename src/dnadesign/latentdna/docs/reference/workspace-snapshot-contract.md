<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[LatentDNA guide](../README.md)

# Workspace Snapshot Contract

`latentdna workspace snapshot --workspace <id|path> --json` summarizes configured inputs and analysis outputs, then writes
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
representation view. Each row records the candidate set memberships, source,
dataset or path, row basis, model name, feature family, modality, sequence
scope, pooling operation, orientation, coordinate space, role, dimensions when
known, materialization status, and freshness status. Planned or retired
candidate views remain visible in this ledger instead of disappearing from
representation comparisons. The same row contract is also embedded in generated
notebook controls so Marimo startup uses precomputed ledger metadata rather
than reopening candidate matrices just to describe row counts or dimensions.

Integrations should consume these public interfaces:

- the study-owned binding file
- the workspace snapshot schema
- the LatentDNA CLI contract
- exported artifact formats

Use those interfaces rather than importing LatentDNA internals or reconstructing workspace semantics from provider modules.
