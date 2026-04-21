## stress_ethanol_cipro_growth Routes

**Last verified:** 2026-04-17

Use this page after the checked-in study status tells you where the record stands.
Use preflight when you need blockers or next-run readiness.
This page keeps the downstream handoff map in one place.

- Status: `uv run ops progress show usr.data-plane.promoter-study-status --json`
- Preflight: `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --json`
- LatentDNA downstream snapshot: `uv run latentdna workspace snapshot --workspace stress_ethanol_cipro_growth --json`
- Snapshot route inventory: `evidence.analysis_surfaces.{densegen,latentdna,cluster}`

### Terminology guardrails

- DenseGen generation plans are biological generation conditions such as `background_only`, `ethanol`, `ciprofloxacin`, and `ethanol_ciprofloxacin`.
- Study lifecycle phases are record-plane state labels such as the current `infer_batch_preparation`; they are not DenseGen generation plans.
- Infer lanes are model-family and dataset-target configs such as `anchor_only_20b` or `anchor_plus_template_7b`; they are not lifecycle phases.
- Route `Plane` values use the repo-wide enum from `ARCHITECTURE.md`. If extra nuance is needed, use `Surface role` rather than inventing a new plane name.

### DenseGen EDA

- Type: `route`
- Plane: `data-plane`
- Surface role: `producer`
- Owner-boundary: `densegen`
- Current state: `parallel_optional`
- Entry artifact: `densegen/study_stress_ethanol_cipro`
- Exit artifact: `evidence.analysis_surfaces.densegen` plus `outputs/plots/current_inventory.json`
- Primary doc/workspace: `src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/README.md`
- First command: `uv run dense plot -c src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/config.yaml`
- Route note: DenseGen owns the producer-side plot taxonomy, current inventory, freshness, and notebook visibility contract for this surface.

### Infer lanes

- Type: `route`
- Plane: `control-plane`
- Surface role: `operator`
- Owner-boundary: `infer`
- Current state: `infer_batch_preparation`
- Entry artifact: `promoter/stress_ethanol_cipro_anchor_set` and `promoter/stress_ethanol_cipro_construct_contexts`
- Exit artifact: checked-in infer lane configs plus the next batch preset declared in `pipeline.yaml`
- Primary doc/workspace: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/README.md`
- First command: `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --json`
- Route note: Infer lanes are execution configs layered on top of the current study phase; they do not replace the study lifecycle record.

### LatentDNA comparison surface

- Type: `route`
- Plane: `data-plane`
- Surface role: `downstream-analysis`
- Owner-boundary: `latentdna`
- Current state: snapshot posture from `latentdna_binding.yaml` plus the LatentDNA workspace-snapshot contract
- Entry artifact: `promoter/stress_ethanol_cipro_anchor_set` and `promoter/stress_ethanol_cipro_construct_contexts`
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
- Appendix deliverables:
  - `appendix_geometry_audit`
  - `appendix_umap_gallery`
- Snapshot attention surfaces: `dataset_overview`, `representation_health_summary`, `design_structure_summary`, `sigma35_ordinal_audit`, `context_robustness_summary`
- Sigma-35 ordinal interpretation for this study follows the reverse-alphabetical promoter ladder on the active subset: `f > e > d > c > b`
- Notebook role: plot-first review surface for pre-assay representation triage; appendix and debug tabs are secondary audit material
- Browser default geometries: the eight canonical anchor/full-context spaces across `intermediate_embedding` and `pooled_logits` for `evo2_7b` and `evo2_20b`
- Interpretation guardrails:
  - do not choose `X` by UMAP aesthetics
  - do not read anchor-local mechanism out of pooled full-sequence vectors
  - do not treat the notebook browser as the authoritative study-status surface
- Route note: use this route for downstream comparison outputs after checking the study record.

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
