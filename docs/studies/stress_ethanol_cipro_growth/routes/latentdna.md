## LatentDNA Route Detail

**Last verified:** 2026-05-17

Use this only after `routes/README.md` selects the LatentDNA comparison
surface. Keep status questions in `../record/status.md` or `ops progress`.

### Surface

- Type: `route-detail`
- Plane: `data-plane`
- Surface role: `downstream-analysis`
- Owner-boundary: `latentdna`
- Current state: `attention`; generated view rows, plots, notebook outputs,
  and candidate-X scorecard are available, but `design_structure_summary` and
  `sigma35_ordinal_audit` remain pending before the full primary review path is
  current.
- Binding file: `docs/studies/stress_ethanol_cipro_growth/bindings/latentdna.yaml`
- Primary doc: `src/dnadesign/latentdna/docs/workflows/stress-ethanol-cipro-representation-comparison.md`
- Workspace: `src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth/README.md`
- Read-only snapshot command: `uv run latentdna workspace snapshot --workspace stress_ethanol_cipro_growth --json --dry-run`
- Publish snapshot command: `uv run latentdna workspace snapshot --workspace stress_ethanol_cipro_growth --json`
- Validation command: `uv run latentdna validate workspace --workspace stress_ethanol_cipro_growth --deep`

### Review Order

- Gate: `representation_health_summary`
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
- Appendix surfaces:
  - `sigma35_centroid_distance_gallery`
  - `native_tf_axis_orientation_audit`
  - `native_regulator_plan_margin_enrichment`
  - `native_regulator_go_bp_plan_margin_enrichment`
  - `appendix_geometry_review`
  - `appendix_umap_gallery`

### Current Representation Choices

- Current working pre-assay `X`: `intermediate_embedding_7b_context_anchor_mean_bidir_concat`
- Conservative DenseGen-plan baseline: `intermediate_embedding_7b_anchor_60bp`
- Strength-standard interpretation lens: `intermediate_embedding_7b_full_context_anchor_mean`
- Preferred infer family: `evo2_7b`
- The preferred infer family is now `evo2_7b`; 20B is a comparison lane, not the default browser gate.
- Sigma-35 ordinal interpretation follows the reverse-alphabetical promoter
  ladder on the active subset: `f > e > d > c > b`.
- Snapshot attention surfaces: `design_structure_summary`, `sigma35_ordinal_audit`

### Appendix Boundaries

- `native_tf_axis_orientation_audit` is the BaeR/CpxR/LexA regulator landmark
  audit over the existing `intermediate_embedding_7b_context_anchor_mean_bidir_concat`
  view. It is not an OPAL input.
- BioCyc GO sidecars support regulator interpretation in appendix enrichment
  surfaces. They are source-backed annotation sidecars, not phenotype labels
  and not OPAL candidate-selection inputs.
- The configured exploratory regulator appendix keeps BaeR/CpxR/LexA landmark
  orientation separate from post-hoc regulator discovery.

### Browser Guardrails

- The notebook is a plot-first review surface for pre-assay representation
  triage; it is not the authoritative study-status surface.
- 7B-first sidecar-backed browser posture is intentional for the current
  review loop.
- Prefer available 7B sequence-view sidecar geometries when reviewing the
  current browser.
- Do not choose `X` by UMAP aesthetics.
- Do not compare absolute UMAP coordinates across population refreshes.
- Do not read anchor-local mechanism out of pooled full-sequence vectors.
- Describe `seq_mean`, `anchor_mean`, and `core60_mean` as token-position
  means over causal Evo2 states in the emitted orientation.
- Treat bidirectional forward/RC context concat as an aggregation strategy, not as native bidirectional encodings.
- Reference hues are workspace-configured and cohort-gated. Do not hardcode a
  generic promoter browser ontology.
