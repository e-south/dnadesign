# USR Datasets Index

This folder contains local USR datasets. Dataset contents are intentionally ignored by git
(`records.parquet`, snapshots, logs, derived overlays). Update this index when datasets are added or
archived so collaborators can discover what exists without committing the data.

## Conventions
- Canonical file: `records.parquet`
- Notes: `meta.md`
- Derived overlays: `_derived/<namespace>.parquet` (namespaced columns only)
- Generated: `.events.log`, `_snapshots/`
- Dataset ids: flat ids such as `mg1655_promoters` are first-class; `namespace/dataset` remains available when it improves disambiguation
- Registry: `registry.yaml` at the datasets root (required for overlays and tracked in repo-owned shared roots because it is the cross-tool namespace contract, not generated run state; byte-stable ordering matters because `usr:registry_hash` is computed from the serialized YAML)

For the shared repo root `src/dnadesign/usr/datasets`, keep `registry.yaml` committed and synced across clones before relying on `usr validate --strict` or SSH sync verification.

## Local Datasets (update as needed)
- `demo/` — demo dataset for CLI examples
- `60bp_dual_promoter_cpxR_LexA/` — canonical densegen dataset (do not edit without review)
- `mg1655_promoters/` — curated wildtype promoter source dataset currently used by construct and promoter-study assembly flows
- `plasmids/` — curated plasmid/template source dataset currently used by construct and promoter-study assembly flows
- `_archive/` — generated or historical datasets (do not hand-edit)
