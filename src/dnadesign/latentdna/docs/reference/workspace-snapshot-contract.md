# Workspace Snapshot Contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-15

`latentdna workspace snapshot --workspace <id|path> --json` publishes the sanctioned study-facing status surface and writes `outputs/status/workspace_snapshot.json`.

Required top-level fields:

- `schema_version`
- `workspace_id`
- `output_root`
- `sources`
- `model_families`
- `canonical_views`
- `deliverables`
- `exports`
- `browser`
- `decision_ladder`
- `last_updated_at`

`decision_ladder` publishes the primary decision surfaces only. Gate
deliverables still appear under `deliverables`, but they are not listed as
decision steps.

Promoter-study tooling is allowed to know only:

- the study-owned binding file
- the workspace snapshot schema
- the LatentDNA CLI contract
- exported artifact formats

Promoter-study tooling must not import LatentDNA internals or reconstruct workspace semantics from provider modules.
