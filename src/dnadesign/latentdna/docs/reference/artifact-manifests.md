# Artifact Manifests

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-11

Each persisted artifact directory contains a `manifest.json` with at least:

- schema version
- artifact kind and id
- workspace id
- command provenance
- recorded inputs and digests
- primary outputs
- row and dimension stats where relevant

Freshness guidance:

- source-backed artifacts should record concrete source provenance paths and digests.
- alignment and export artifacts now record path-backed inputs so deliverable status can detect stale upstream inputs rather than reporting unknown freshness.
- artifacts that omit path-backed provenance still degrade to `attention` rather than silently claiming freshness.

See also:

- [deliverable-contract.md](deliverable-contract.md)
- [performance-budgets.md](performance-budgets.md)
