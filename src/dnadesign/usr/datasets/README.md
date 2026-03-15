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
- Registry: `registry.yaml` at the datasets root (required for overlays)

## Local Datasets (update as needed)
- `demo/` — demo dataset for CLI examples
- `60bp_dual_promoter_cpxR_LexA/` — canonical densegen dataset (do not edit without review)
- `mg1655_promoters/` — optional construct demo/control promoter dataset created by `construct seed promoter-swap-demo`
- `plasmids/` — optional construct demo plasmid dataset created by `construct seed promoter-swap-demo`
- `_archive/` — generated or historical datasets (do not hand-edit)
