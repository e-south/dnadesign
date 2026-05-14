## stress_ethanol_cipro_growth Routes

**Last verified:** 2026-05-14

Use this page after the checked-in study status tells you where the record stands.
Use preflight when you need blockers or next-run readiness.
This page keeps the downstream handoff map in one place.

- Status: `uv run ops progress show usr.data-plane.promoter-study-status --json`
- Preflight: `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --command-timeout-seconds 30 --json`
- LatentDNA downstream snapshot: `uv run latentdna workspace snapshot --workspace stress_ethanol_cipro_growth --json`
- Snapshot route inventory: `evidence.analysis_surfaces.{densegen,latentdna,cluster}`

### Terminology guardrails

- DenseGen generation plans are biological generation conditions such as `background_only`, `ethanol`, `ciprofloxacin`, and `ethanol_ciprofloxacin`.
- Study lifecycle phases are record-plane state labels such as the current `latentdna_reference_normalization_audit`; they are not DenseGen generation plans.
- Infer lanes are model-family and dataset-target configs such as `anchor_only_20b` or `anchor_plus_template_7b`; they are not lifecycle phases.
- Route `Plane` values use the repo-wide enum from `ARCHITECTURE.md`. If extra nuance is needed, use `Surface role` rather than inventing a new plane name.

### DenseGen EDA

- Type: `route`
- Plane: `data-plane`
- Surface role: `producer`
- Owner-boundary: `densegen`
- Current state: `attention`; source rows are ready, but operator-visible plot artifacts are stale in the current snapshot
- Entry artifact: `densegen_prom_eth_cip_source`
- Exit artifact: `evidence.analysis_surfaces.densegen` plus `outputs/plots/current_inventory.json`
- Primary doc/workspace: `src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/README.md`
- First command: `uv run dense plot -c src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/config.yaml`
- Route note: DenseGen owns the producer-side plot taxonomy, current inventory, freshness, and notebook visibility contract for this surface.

### Infer lanes

- Type: `route`
- Plane: `control-plane`
- Surface role: `operator`
- Owner-boundary: `infer`
- Current state: `complete` for supported Evo2 7B sequence-view sidecars
- Entry artifact: `usr_prom_eth_cip_anchor`, `construct_prom_eth_cip_context`,
  `construct_prom_eth_cip_reference_core60`, and
  `construct_prom_eth_cip_reference_contexts`
- Exit artifact: dataset-local `_derived/infer/` sidecars plus checked-in infer lane configs
- Primary doc/workspace: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/README.md`
- First command: `uv run ops runbook fill-infer --study-dir docs/studies/stress_ethanol_cipro_growth --no-submit`
- Route note: Infer lanes are execution configs layered on top of the current
  study phase; they do not replace the study lifecycle record. The supported
  Evo2 7B lanes now plan no runnable GPU work. Notify runbooks remain the
  historical execution surfaces for one USR event stream per lane, including
  the split reference core60, reference-context-forward, and
  reference-context-reverse lanes.

### Construct anchor/context refresh

- Type: `route`
- Plane: `data-plane`
- Surface role: `consolidation`
- Owner-boundary: `usr` plus `construct`
- Current state: `complete`
- Entry artifact: `densegen_prom_eth_cip_source`, `usr_promoter_references`, and `usr_sfxi_pdual10_densegen_promoters`
- Exit artifact: `construct_prom_eth_cip_reference_core60`, `construct_prom_eth_cip_reference_contexts`, `usr_prom_eth_cip_anchor`, and `construct_prom_eth_cip_context`
- Primary workspace: `src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10`
- First command: `uv run construct workspace run-project --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 --project reference_core60 --dry-run --format json`
- Route note: Construct first derives fail-fast `analysis_window` reference views from native GenBank annotations, then emits paired forward and reverse-complement reference contexts. USR then converges DenseGen 60 bp anchors, native reference inserts, SFXI pDual rows, and reference core60 rows into the shared anchor dataset. After merge, `dnadesign.usr.scripts.materialize_promoter_anchor_sequence_views` writes one `construct_insert` anchor-only sequence view per merged row; native or designed exact-60 rows are not relabeled as `analysis_window`. The shared paired forward and reverse-complement pDual context refresh follows from that anchor handoff.
- Native regulator landmark extension: `usr_regulondb_native_promoters` has populated regulatory interactions, RegulonDB parent metadata is projected to `usr_regulondb_native_promoter_core60`, and that core60 source is merged into `usr_prom_eth_cip_anchor` before the existing `forward_anchor_window` project emits paired pDual contexts into `construct_prom_eth_cip_context`. The BaeR/CpxR/LexA landmark audit uses the 3180 unambiguous parent-resolved native core60 rows; one duplicate core60 sequence collapses two NhaR-only native parents and is excluded from the single-parent regulator overlay. Do not create a separate `construct_prom_eth_cip_native_tf_contexts` universe for this audit.

### LatentDNA comparison surface

- Type: `route`
- Plane: `data-plane`
- Surface role: `downstream-analysis`
- Owner-boundary: `latentdna`
- Current state: `current`; the workspace config supports the BaeR/CpxR/LexA regulator landmark overlay, and generated LatentDNA view rows/plots/notebook outputs have been refreshed after the lineage-metadata change
- Entry artifact: `usr_prom_eth_cip_anchor`, `construct_prom_eth_cip_context`, `construct_prom_eth_cip_reference_core60`, and `construct_prom_eth_cip_reference_contexts`
- Exit artifact: published LatentDNA workspace snapshot plus sanctioned comparison deliverables and the `latent_geometry_browser` notebook
- Binding file: `docs/studies/stress_ethanol_cipro_growth/latentdna_binding.yaml`
- Primary doc: `src/dnadesign/latentdna/docs/workflows/promoter-study-representation-comparison.md`
- Workspace: `src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth/README.md`
- First command: `uv run latentdna workspace snapshot --workspace stress_ethanol_cipro_growth --json`
- Snapshot artifact: the path declared by `latentdna_binding.yaml`
- Follow-up validation: `uv run latentdna validate workspace --workspace stress_ethanol_cipro_growth --deep`
- Gate:
  1. `representation_health_summary`
- Primary review path:
  1. `dataset_overview`
  2. `design_structure_summary`
  3. `sigma35_ordinal_audit`
  4. `context_robustness_summary`
  5. `candidate_decision_frontier`
  6. `candidate_x_selection_scorecard`
- Companion visuals:
  - `balanced_design_family_margin_gallery`
  - `sigma35_margin_ladder_gallery`
  - `sigma35_stress_margin_gallery`
  - `context_pair_summary`
  - `reference_to_plan_centroid_heatmap`
  - `reference_standard_strength_audit`
- Pre-specified regulator landmark overlay:
  - `native_tf_axis_orientation_audit` is the stable artifact id for the BaeR/CpxR/LexA regulator landmark audit. It reuses the existing `intermediate_embedding_7b_context_anchor_mean_bidir_concat` view after the shared anchor/context quota is expanded with `usr_regulondb_native_promoter_core60`; it requires `usr_regulondb_native_promoters/_relations/regulatory_interactions.parquet` and is not an OPAL input.
- Configured exploratory regulator appendix:
  - `native_regulator_plan_margin_enrichment` is specified in `src/dnadesign/latentdna/docs/dev/native-regulator-plan-margin-enrichment.md`. It tests source-backed RegulonDB regulator enrichments in synthetic-plan margins and fixed 5%/10% native tails, while keeping the existing BaeR/CpxR/LexA landmark audit separate from post-hoc regulator discovery. It now emits contract tables plus a static appendix plot surfaced in the `latent_geometry_browser` plot-review notebook panel.
- RegulonDB regulator functional terms:
  - BioCyc KB 29.6 SmartTable GO terms are projected onto `usr_regulondb_native_promoters` through `src/dnadesign/usr/scripts/project_regulondb_functional_annotations.py`.
  - The method contract is `src/dnadesign/usr/docs/reference/regulondb-functional-annotations.md`.
  - The public source descriptors live in `dnadesign-data` as `biocyc_29_6_smarttable_regulator_go_terms` and `biocyc_29_6_smarttable_regulator_go_coverage`; downstream code should use `dnadesign_data.catalog.sources.resolve_source_record` or `dnadesign-data-sources resolve`, not private path guesses.
  - Materialized sidecars are `_relations/regulator_go_terms.parquet`, `_relations/promoter_regulator_go_terms.parquet`, and `_relations/regulator_go_coverage.parquet`. They support appendix interpretation only and are not OPAL/candidate-selection inputs.
- Appendix deliverables:
  - `sigma35_centroid_distance_gallery`
  - `native_tf_axis_orientation_audit`
  - `native_regulator_plan_margin_enrichment`
  - `appendix_geometry_review`
  - `appendix_umap_gallery`
- Snapshot attention surfaces: none for LatentDNA decision deliverables after the native lineage metadata/config update
- Snapshot ok primary surfaces: `dataset_overview`, `design_structure_summary`, `sigma35_ordinal_audit`, `context_robustness_summary`, `candidate_decision_frontier`, `candidate_x_selection_scorecard`
- Snapshot ok appendix surfaces: `native_tf_axis_orientation_audit`, `appendix_geometry_review`, `appendix_umap_gallery`
- Current working pre-assay `X`: `intermediate_embedding_7b_context_anchor_mean_bidir_concat`
- Conservative DenseGen-plan baseline: `intermediate_embedding_7b_anchor_60bp`
- Strength-standard interpretation lens: `intermediate_embedding_7b_full_context_anchor_mean`
- Sigma-35 ordinal interpretation for this study follows the reverse-alphabetical promoter ladder on the active subset: `f > e > d > c > b`
- Notebook role: plot-first review surface for pre-assay representation triage; appendix and debug tabs are secondary audit material
- Browser default geometry layout: candidate grid over the six available 7B
  intermediate sequence-view sidecar geometries with persisted UMAPs: anchor
  construct-insert `seq_mean`, forward full-context `seq_mean`, forward
  full-context `anchor_mean`, controlled equal-block bidirectional forward/RC
  `anchor_mean`, reverse-complement full-context `seq_mean`, and
  reverse-complement full-context `anchor_mean`. Reference core60 and
  reference-context views remain hidden audit geometries until promoted to a
  study-facing projection layout. Output-layer mean views are mean-pooled
  per-token output-logits diagnostics using the same pooling scopes, and
  log-likelihoods are scalar diagnostics; neither family is current decision
  geometry.
- Browser hue controls are workspace-configured, not promoter hardcoded. The
  main `Hue` menu colors population rows by metadata carried on materialized
  view rows, including `source_family`, `selection_basis`,
  `promoter_standard__collection_id`, and continuous
  `promoter_standard__strength_value_numeric` when those columns are present.
  Reference overlays use separate `Reference labels`, `Reference annotations`,
  and `Reference hue` controls. `Reference hue` options are conditional on the
  selected reference cohort and visible plot frame: SFXI-scored archive rows
  expose `SFXI score`, `SFXI logic fidelity`, and `SFXI effect scaled`;
  RegulonDB native core60 and BaeR/CpxR/LexA TF-axis rows expose
  `Native TF bin`; Anderson and W collection rows expose `Reference strength`;
  native MG1655 GenBank and spyP/sulAp overlays do not expose unsupported
  reference hues.
- Interpretation guardrails:
  - do not choose `X` by UMAP aesthetics
  - do not compare absolute UMAP coordinates across population refreshes; seeded UMAPs still refit when rows are appended
  - do not read anchor-local mechanism out of pooled full-sequence vectors
  - describe `seq_mean`, `anchor_mean`, and `core60_mean` as token-position
    means over causal Evo2 states in the emitted orientation, not as native bidirectional encodings
  - do not let reference-neighbor behavior replace synthetic internal-structure gates
  - do not treat Anderson and W numeric strengths as one pooled biological scale
  - do not treat the notebook browser as the authoritative study-status surface
- Route note: use this route for downstream comparison outputs after checking
  the study record. This is a 7B-first sidecar-backed browser posture; the
  available 7B sequence-view sidecar geometries are the current review basis,
  and the preferred infer family is now `evo2_7b`.

### Cluster exploration

- Type: `route`
- Plane: `data-plane`
- Surface role: `downstream-analysis`
- Owner-boundary: `cluster`
- Current state: `planned`
- Entry artifact: `context_robustness_summary`
- Exit artifact: study-owned cluster workspace or results root once this route is configured
- Primary doc/workspace: `src/dnadesign/cluster/docs/workflows/exploratory-clustering.md`
- First command: `uv run ops catalog show cluster.downstream.exploratory-clustering`

### OPAL campaigns

- Type: `route`
- Plane: `control-plane`
- Surface role: `decision`
- Owner-boundary: `opal`
- Current state: `not_configured`
- Entry artifact: study-owned feature bundle chosen outside LatentDNA
- Exit artifact: OPAL-ready feature bundle plus campaign config when a downstream decision is made
- Primary doc/workspace: `src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md`
- First command: `uv run ops catalog show opal.downstream.usr-infer-x-active-learning`
- Boundary note: LatentDNA can narrow the choice of `X`, but it does not own the downstream OPAL handoff decision.
