## stress_ethanol_cipro_growth Routes

**Last verified:** 2026-04-15

Open this page after status answers `where is the study now?` and preflight
answers `what blocks execution here?`

Treat `status.md` as the study snapshot only. Keep owner-tool commands,
cleanup steps, and notebook or deliverable run loops on this page or the
tool-local workflow docs.

- Status: `uv run ops progress show usr.data-plane.promoter-study-status --json`
- Preflight: `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --json`
- This page: downstream owner handoff into DenseGen, Construct, Infer,
  LatentDNA, Cluster, and OPAL
- Snapshot route inventory: `evidence.analysis_surfaces.{densegen,latentdna,cluster}`
  in the status JSON for plot ids, artifact roots, notebook paths, and command
  templates

### DenseGen EDA

- Current state: `parallel_optional`; shared DenseGen source dataset is current at `157160` rows.
- Owner tool: `densegen`
- Entry artifact: `densegen/study_stress_ethanol_cipro`
- Primary doc/workspace: `src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/README.md`
- First command: `uv run dense plot -c src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/config.yaml`
- Status JSON route: `evidence.analysis_surfaces.densegen`
- Default plot ids: `dataset_source_inventory`, `dataset_metadata_heatmap`,
  `stage_a_summary`, `placement_map`, `run_health`, `tfbs_usage`
- Optional plot id: `dense_array_video_showcase`
- Live artifact paths: `src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/outputs/plots/plot_manifest.json`
  and `src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/outputs/notebooks/densegen_run_overview.py`
- Configured/planned/not configured: `configured`

### Construct lineage

- Current state: `complete`; the shared construct-context dataset is current at `157164` rows.
- Owner tool: `construct`
- Entry artifact: `promoter/stress_ethanol_cipro_anchor_set`
- Primary doc/workspace: `src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10/runbook.md`
- First command: `uv run construct workspace doctor --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10`
- Configured/planned/not configured: `configured`

### Infer lanes

- Current state: `infer_batch_preparation`; shared handoff datasets are ready and the next execution gate is preflight for the lane-specific Infer presets.
- Owner tool: `infer`
- Entry artifact: `promoter/stress_ethanol_cipro_anchor_set` and `promoter/stress_ethanol_cipro_construct_contexts`
- Primary doc/workspace: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/README.md`
- First command: `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --json`
- Configured/planned/not configured: `configured`

### LatentDNA atlas

- Current state: `attention`; the study-bound workspace validates on the canonical `outputs/` root, the browser notebook now has the geometry switchboard plus context-audit controls, and the open gap is still the missing `x2_primary_20b` export bundle rather than notebook or plot materialization.
- Owner tool: `latentdna`
- Entry artifact: `promoter/stress_ethanol_cipro_anchor_set` and `promoter/stress_ethanol_cipro_construct_contexts`
- Primary doc: `src/dnadesign/latentdna/docs/workflows/promoter-study-latent-atlas.md`
- Workspace: `src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth/README.md`
- First command: `uv run latentdna validate workspace --workspace src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth --deep`
- Canonical artifact root: `src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth/outputs`
- Invalid legacy root: `src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth/outputs/latentdna`
- Safe cleanup command: `uv run latentdna workspace refresh --workspace src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth --target legacy --dry-run`
- Status JSON route: `evidence.analysis_surfaces.latentdna`
- Plot ids: `atlas_2x2_intermediate_main`, `control_pca_explained_variance_curve`,
  `context_shift_primary_distribution`, `drag_qc_distribution`,
  `context_shift_vs_drag_primary`, `cluster_correspondence_primary`,
  `agreement_7b_vs_20b_summary`
- Materialized now: `atlas_2x2_intermediate_main`, `atlas_2x3_model_family`,
  `control_pca_explained_variance_curve`,
  `context_shift_primary_distribution`, `drag_qc_distribution`,
  `context_shift_vs_drag_primary`, `agreement_7b_vs_20b_summary`,
  `cluster_correspondence_primary`, the browser notebook, and
  `outputs/scalars/context_audit_20b/table.parquet`
- Deliverables currently declared in the workspace include:
  `atlas_2x2_intermediate_main`, `geometry_switchboard_20b`,
  `control_pca_explained_variance_curve`, `drag_qc`,
  `context_shift_primary`, `context_audit_primary_20b`, `agreement_7b_vs_20b`,
  `cluster_correspondence_primary`, `x0_primary_20b`, `x1_primary_20b`,
  `x2_primary_20b`, `x3_ablation_7b`
- Still missing: export bundles `x0_primary_20b`, `x1_primary_20b`,
  `x2_primary_20b`, and `x3_ablation_7b`; treat `x2_primary_20b` as the open
  export gate rather than a live downstream handoff
- Interpretation boundary: the saved study artifacts support pooled
  expanded-context representation analysis, not token-level hidden states,
  per-position logits, or anchor-only slices inside the 1 kb construct
- Planned next analysis pack, not yet materialized: reference-centric
  `d_E`/`d_C` similarity maps against WT controls and synthetic centroids,
  paired anchor-to-full-context shift summaries, geometry-preservation
  scorecards, and grouped-CV or precision-at-K leaderboards
- Decision rule for the next slice: `z20_60` and `z20_1k_seq` intermediate
  embeddings are the primary candidate space, pooled logits stay as
  benchmarks, per-base likelihood stays a scalar side channel, and
  `z20_1k_anchor`, `drag20`, and Leiden correspondence stay QC until the new
  reference and prediction plots exist
- Review loop: `workspace refresh` -> `deliverable status` or `recipe run` -> `notebook generate` (allowed to be `attention`) -> `notebook smoke` -> `marimo run`
- Browser control-plane artifact: `outputs/notebooks/browser/controls.json`
- Context-audit decision artifact: `outputs/scalars/context_audit_20b/table.parquet` plus the browser `context_audit` summary
- Real-host note: the atlas path now succeeds end to end, but the full-view PCA lanes behind `context_shift_primary`, `control_pca_explained_variance_curve`, and `x2_primary_20b` can require `--allow-memory-overage` on a 16 GiB workstation.
- Real-host note: `cluster_correspondence_primary` is now routed through aligned reduced views plus explicit neighbor graphs; the raw aligned Leiden path on `anchor_ctx_20b` is not a viable operator flow on a 16 GiB workstation.
- Artifact path roots: `outputs/plots/<plot-id>/manifest.json`,
  `outputs/notebooks/<notebook-id>/notebook.py`, and
  `outputs/<artifact-kind>/<artifact-id>/manifest.json`
- Configured/planned/not configured: `configured`

### Cluster exploration

- Current state: `planned`; no study-owned cluster results root is configured yet.
- Owner tool: `cluster`
- Entry artifact: `promoter/stress_ethanol_cipro_feature_matrix` or a later explicit latent export once a canonical export bundle is materialized (current gap: `x2_primary_20b`)
- Primary doc/workspace: `src/dnadesign/cluster/docs/workflows/exploratory-clustering.md`
- First command: `uv run ops catalog show cluster.downstream.exploratory-clustering`
- Status JSON route: `evidence.analysis_surfaces.cluster`
- Workspace example only: `src/dnadesign/cluster/workspaces/promoter_clusters_v1/config.yaml`
- Artifact path template: `<results-root>/<alias>/{fits,umap,analysis,sweeps}/<run-slug>/`
- Common plot families once materialized: `umap_png`,
  `composition_proportions_png`, `diversity_png`, `numeric_violin_png`,
  `resolution_sweep_png`
- Configured/planned/not configured: `planned`

### OPAL campaigns

- Current state: `not configured`; no study-owned OPAL campaign config is checked in yet.
- Owner tool: `opal`
- Entry artifact: `promoter/stress_ethanol_cipro_feature_matrix` or a later explicit latent export once a canonical export bundle is materialized (current gap: `x2_primary_20b`)
- Primary doc/workspace: `src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md`
- First command: `uv run ops catalog show opal.downstream.usr-infer-x-active-learning`
- Configured/planned/not configured: `not configured`
