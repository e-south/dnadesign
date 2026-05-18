## stress_ethanol_cipro_growth Routes

**Last verified:** 2026-05-17

Use this page after the checked-in study status tells you where the record stands. Keep this file as the one-hop handoff map; put downstream detail in focused files under `routes/`.

- Status: `uv run ops progress show studies.stress-ethanol-cipro-growth.status --json`
- Preflight: `uv run ops progress show studies.stress-ethanol-cipro-growth.preflight --scope next --command-timeout-seconds 30 --json`
- LatentDNA detail: [`routes/latentdna.md`](routes/latentdna.md)
- OPAL detail: [`routes/opal.md`](routes/opal.md)
- Snapshot route inventory: `evidence.analysis_surfaces.{densegen,latentdna,cluster}`

### Terminology guardrails

- DenseGen generation plans are biological generation conditions such as `background_only`, `ethanol`, `ciprofloxacin`, and `ethanol_ciprofloxacin`.
- OPAL campaigns are downstream objectives: ethanol factor, ciprofloxacin factor, and AND. AND is not a synonym for every `ethanol_ciprofloxacin` DenseGen row.
- OPAL reads an `opal_candidate_feature_table`, not just a matrix. The materialized table is the dense generated promoter subset in `usr_prom_eth_cip_opal_candidates` with X column `latentdna__evo2_7b__context_anchor_mean_bidir_concat`.
- SFXI state order for these campaigns is `[00, 10, 01, 11]`.
- Study lifecycle phases are record-plane state labels such as the current `latentdna_reference_normalization_audit`; they are not DenseGen generation plans.
- Infer lanes are model-family and dataset-target configs such as `anchor_only_20b` or `anchor_plus_template_7b`; they are not lifecycle phases.
- Route `Plane` values use the repo-wide enum from `ARCHITECTURE.md`. If extra nuance is needed, use `Surface role` rather than inventing a new plane name.

### DenseGen EDA

- Type: `route`
- Plane: `data-plane`
- Surface role: `producer`
- Owner-boundary: `densegen`
- Current state: `attention`; source rows are ready, but operator-visible plot
  artifacts are stale in the current snapshot
- Entry artifact: `densegen_prom_eth_cip_source`
- Exit artifact: `evidence.analysis_surfaces.densegen` plus
  `outputs/plots/current_inventory.json`
- Primary doc/workspace: `src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/README.md`
- First command: `uv run dense plot -c src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/config.yaml`
- Route note: DenseGen owns the producer-side plot taxonomy, current inventory,
  freshness, and notebook visibility contract for this surface.

### Infer lanes

- Type: `route`
- Plane: `control-plane`
- Surface role: `operator`
- Owner-boundary: `infer`
- Current state: `complete` for supported Evo2 7B sequence-view sidecars
- Entry artifact: `usr_prom_eth_cip_anchor`, `construct_prom_eth_cip_context`,
  `construct_prom_eth_cip_reference_core60`, and
  `construct_prom_eth_cip_reference_contexts`
- Exit artifact: dataset-local `_derived/infer/` sidecars plus checked-in infer
  lane configs
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
- Entry artifact: `densegen_prom_eth_cip_source`,
  `usr_promoter_references`, and `usr_sfxi_pdual10_densegen_promoters`
- Exit artifact: `construct_prom_eth_cip_reference_core60`,
  `construct_prom_eth_cip_reference_contexts`, `usr_prom_eth_cip_anchor`,
  and `construct_prom_eth_cip_context`
- Primary workspace: `src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10`
- First command: `uv run construct workspace run-project --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 --project reference_core60 --dry-run --format json`
- Route note: Construct owns the reference-view and context-refresh lineage.
  For native regulator audit details, use the checked-in status note and
  LatentDNA route detail instead of extending this map.

### LatentDNA comparison surface

- Type: `route`
- Plane: `data-plane`
- Surface role: `downstream-analysis`
- Owner-boundary: `latentdna`
- Current state: `attention`
- Gate: `representation_health_summary`
- Primary review path: `dataset_overview`, `design_structure_summary`,
  `sigma35_ordinal_audit`, `context_robustness_summary`,
  `candidate_decision_frontier`, `candidate_x_selection_scorecard`
- Snapshot attention surfaces: `design_structure_summary`, `sigma35_ordinal_audit`
- Companion visuals: `balanced_design_family_margin_gallery`,
  `sigma35_margin_ladder_gallery`, `sigma35_centroid_distance_gallery`,
  `sigma35_stress_margin_gallery`, `context_pair_summary`,
  `appendix_umap_gallery`
- Current working pre-assay `X`: `intermediate_embedding_7b_context_anchor_mean_bidir_concat`
- Browser posture: 7B-first sidecar-backed browser posture over the available 7B sequence-view sidecar geometries; preferred infer family is now `evo2_7b`. Interpret pooled features as token-position means, not as native bidirectional encodings.
- Entry artifact: `usr_prom_eth_cip_anchor`, `construct_prom_eth_cip_context`,
  `construct_prom_eth_cip_reference_core60`, and
  `construct_prom_eth_cip_reference_contexts`
- Exit artifact: published LatentDNA workspace snapshot plus sanctioned
  comparison deliverables and the `latent_geometry_browser` notebook
- Detail: [`routes/latentdna.md`](routes/latentdna.md)
- First command: `uv run latentdna workspace snapshot --workspace stress_ethanol_cipro_growth --json`
- Route note: use this route for downstream comparison outputs after checking
  the study record. LatentDNA detail stays in `routes/latentdna.md`.

### Cluster exploration

- Type: `route`
- Plane: `data-plane`
- Surface role: `downstream-analysis`
- Owner-boundary: `cluster`
- Current state: `planned`
- Entry artifact: `context_robustness_summary`
- Exit artifact: study-owned cluster workspace or results root once this route
  is configured
- Primary doc/workspace: `src/dnadesign/cluster/docs/workflows/exploratory-clustering.md`
- First command: `uv run ops catalog show cluster.downstream.exploratory-clustering`

### OPAL campaigns

- Type: `route`
- Plane: `control-plane`
- Surface role: `decision`
- Owner-boundary: `opal`
- Current state: `candidate_table_materialized_pre_assay`
- Entry artifact: `usr_prom_eth_cip_opal_candidates`
- Exit artifact: campaign-owned OPAL ledgers under each `outputs/ledger/`
- Primary doc/workspace: `src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md`
- Batch-0 selector: `src/dnadesign/studies/studies/stress_ethanol_cipro_growth/opal_batch0/`
- Candidate table role: `opal_candidate_feature_table`
- Candidate table X: `latentdna__evo2_7b__context_anchor_mean_bidir_concat`
- Detail: [`routes/opal.md`](routes/opal.md)
- First command: `uv run opal validate -c src/dnadesign/opal/campaigns/stress_eth_cip_ethanol_rf_sfxi_topn/configs/campaign.yaml`
- Per-ID provenance command: `uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_batch0.provenance --config src/dnadesign/studies/studies/stress_ethanol_cipro_growth/opal_batch0/sampling.yaml --id <candidate_id>`
- Boundary note: LatentDNA can narrow the choice of `X`, but it does not own
  the downstream OPAL handoff decision. Batch 0 is a pre-assay seed; after
  labels exist, OPAL owns label-source validation, training, scoring, active
  selection, and campaign ledgers.
