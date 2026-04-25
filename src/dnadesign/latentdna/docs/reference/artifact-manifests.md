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
- USR overlay provenance keeps namespace-level `overlay` paths explicit. Legacy manifests may still carry `overlay_part` entries with per-part content digests. Regenerated manifests may instead carry explicit `overlay_ledger` entries when the source dataset has an opt-in `digest_ledger.json` contract for that namespace.
- The ledger-backed contract is explicit rather than implicit: `overlay` inventory digests still detect add/remove/rename drift, while `overlay_ledger` digests collapse per-part content tracking into one small sidecar file. Artifacts only receive that faster path after the source overlay ledger is written and the artifact is regenerated.
- alignment and export artifacts now record path-backed inputs so deliverable status can detect stale upstream inputs rather than reporting unknown freshness.
- managed artifact-to-artifact dependencies record the upstream `manifest.json` path and digest, then recurse into upstream freshness; this keeps workspace-owned immutable artifact contracts explicit without rehashing large managed payload files on every status refresh.
- artifacts that omit path-backed provenance still degrade to `attention` rather than silently claiming freshness.

See also:

- [deliverable-contract.md](deliverable-contract.md)
- [performance-budgets.md](performance-budgets.md)
