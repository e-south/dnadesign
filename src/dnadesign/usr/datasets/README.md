# USR Datasets Index

This folder contains local USR datasets. Dataset contents are intentionally ignored by git
(`records.parquet`, snapshots, logs, derived overlays). Update this index when datasets are added or
archived so collaborators can discover what exists without committing the data.

## Conventions
- Canonical file: `records.parquet`
- Notes: `meta.md`
- Derived overlays: `_derived/<namespace>.parquet` (namespaced columns only)
- Generated: `.events.log`, `_snapshots/`
- Non-canonical workflow artifacts such as `batch_results/`, `outputs/`, and `runs/` may exist in historical datasets; treat them as generated side artifacts, not part of the cross-tool dataset API.
- Dataset ids: flat ids such as `mg1655_promoters` are first-class; `namespace/dataset` remains available when it improves disambiguation
- Registry: `registry.yaml` at the datasets root (required for overlays and tracked in repo-owned shared roots because it is the cross-tool namespace contract, not generated run state; byte-stable ordering matters because `usr:registry_hash` is computed from the serialized YAML)

For the shared repo root `src/dnadesign/usr/datasets`, keep `registry.yaml` committed and synced across clones before relying on `usr validate --strict` or SSH sync verification.

## Local Datasets (update as needed)
- Flat ids:
  - `demo/` — demo dataset for CLI examples
  - `mg1655_promoters/` — curated wildtype promoter source dataset used by construct and promoter-study assembly flows
  - `plasmids/` — curated plasmid/template source dataset used by construct and promoter-study assembly flows
- Namespace-qualified ids:
  - `densegen/demo_sampling_baseline/`
  - `densegen/study_stress_ethanol_cipro/`
  - `promoter/stress_ethanol_cipro_anchor_set/`
  - `promoter/stress_ethanol_cipro_construct_contexts/`
- Archive root:
  - `archived/` — canonical location for archived datasets and archive buckets; keep archived material here instead of the live dataset-id namespace
- Archived datasets:
  - `archived/60bp_dual_promoter_cpxR_LexA/` — archived enriched 60 bp dual-promoter dataset retaining infer, cluster, and OPAL metadata
- Archived artifact buckets:
  - `archived/promoter_misc_pytorch/` — legacy promoter-focused PyTorch batch artifacts (`.pt`, summary, and progress YAML files) moved down from the old top-level USR archive surface
