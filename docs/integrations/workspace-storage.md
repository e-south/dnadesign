---
doc_id: workspace-storage-contract
surface: integration-contract
owner: dnadesign-maintainers
last_verified: 2026-08-26
---

## External workspace storage

DNA Design tools may keep private or large workspace instances outside public
Git checkouts while retaining their existing tool-owned workspace schemas.
`dnadesign.workspace-storage/v1` is the neutral outer envelope for that
location. It does not interpret a LatentDNA, OPAL, Cruncher, or USR workspace.

### Boundary

- The producing tool owns workspace contents, behavior, and schema evolution.
- The storage envelope owns stable identity, producer provenance, retention
  posture, and exact file closure.
- Research Studies owns scientific questions, decisions, and accepted evidence.
- Public repositories contain reusable behavior and small explicit demos, not
  private runtime instances.

The manifest must be named `workspace.storage.json` and live at the explicit
workspace root. It declares:

- `workspace_id`;
- `owner_repository` and `owner_tool`;
- `workspace_schema` and `workspace_schema_version`;
- `producer_revision`;
- `storage_class`: `authoritative`, `reproducible`, `cache`, or `cold`;
- a compatible `retention_policy`;
- `inputs` and `artifacts` as confined relative paths plus lowercase SHA-256 digests;
- optional `original_execution_path` as provenance only;
- `demo`, which is the sole opt-in for a small workspace inside a Git checkout.

### Validation

```bash
uv run dnadesign-workspace-storage validate /absolute/path/to/workspace --json
```

Validation fails before tool execution when:

- JSON keys are duplicated, missing, or unsupported;
- schema, storage class, or retention values are unknown;
- a resource path is absolute, escapes the workspace, or is declared twice;
- a declared file is missing or its digest differs;
- a non-demo workspace resolves inside any Git checkout.

There is no default workspace root and no implicit fallback search. Callers pass
an exact path and may operate only on the verified result.

### Adoption sequence

1. Add and validate the envelope around one existing workspace without moving it.
2. Copy that workspace to external storage and verify every declared digest.
3. Change the tool invocation to the explicit external path.
4. Run the tool-owned acceptance checks.
5. Retire the embedded copy only after parity and retention review.

Large LatentDNA, USR, Cruncher, and historical workspace trees must migrate as
separate owner-reviewed slices. The envelope does not authorize a bulk move.
