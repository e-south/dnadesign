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
- Dataset ids: active shared study datasets use flat owner-first ids such as `densegen_prom_eth_cip_source`; use `root_kind`, `owner_tool`, overlays, and study metadata for provenance instead of nested tool-routing paths.
- Namespace-qualified ids remain available for non-active or local disambiguation cases; `archived/` is the only special top-level bucket in this shared root.
- Registry: `registry.yaml` at the datasets root (required for overlays and tracked in repo-owned shared roots because it is the cross-tool namespace contract, not generated run state; byte-stable ordering matters because `usr:registry_hash` is computed from the serialized YAML)

For the shared repo root `src/dnadesign/usr/datasets`, keep `registry.yaml` committed and synced across clones before relying on `usr validate --strict` or SSH sync verification.

## Local Datasets (update as needed)
- Flat ids:
  - `densegen_demo_sampling_baseline/` — DenseGen-owned demo sampling baseline output dataset
  - `densegen_prom_eth_cip_source/` — DenseGen-owned promoter ethanol/ciprofloxacin source dataset
  - `densegen_study_constitutive_sigma_panel/` — DenseGen-owned constitutive sigma-panel study output dataset
  - `usr_demo_cli_examples/` — USR-owned CLI example dataset for local walkthroughs
  - `usr_mg1655_promoter_controls/` — USR-curated MG1655 promoter control anchors used by construct and promoter-study assembly flows
  - `usr_pdual10_plasmid_template/` — USR-curated pDual-10 plasmid template record used by construct context expansion
  - `usr_prom_eth_cip_anchor/` — USR-owned merged promoter ethanol/ciprofloxacin anchor handoff
  - `construct_prom_eth_cip_context/` — Construct-owned promoter ethanol/ciprofloxacin context handoff
- Archive root:
  - `archived/` — canonical location for archived datasets and archive buckets; keep archived material here instead of the live dataset-id namespace
- Archived datasets:
  - `archived/60bp_dual_promoter_cpxR_LexA/` — archived enriched 60 bp dual-promoter dataset retaining infer, cluster, and OPAL metadata
- Archived artifact buckets:
  - `archived/promoter_misc_pytorch/` — legacy promoter-focused PyTorch batch artifacts (`.pt`, summary, and progress YAML files) moved down from the old top-level USR archive surface
