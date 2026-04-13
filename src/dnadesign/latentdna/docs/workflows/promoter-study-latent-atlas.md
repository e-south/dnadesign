# Promoter-Study Latent Atlas

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-10

**Type:** workflow
**Plane:** downstream-tool
**Owner-boundary:** latentdna
**Registry-id:** latentdna.promoter-study.latent-atlas

`latentdna` owns downstream latent analysis artifacts after vector-bearing USR or file-backed sources already exist. It does not run infer and it does not train OPAL models.

The tracer-bullet route below is implemented and fixture-covered. The active promoter-study path should still be treated as pressure-path guidance until `view materialize` and sampled projection fitting are re-verified end to end on the live study after any upstream USR overlay-scan changes.

### First tracer-bullet path

```bash
# Scaffold a workspace from the promoter-study starter template.
uv run latentdna workspace init \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas \
  --template landmark_atlas_committee \
  --from-study-dir docs/studies/stress_ethanol_cipro_growth

# Validate static config references plus the live study-bound source schema before any artifact writes.
uv run latentdna validate workspace \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas \
  --deep

# Freeze one source row ledger plus metadata companion up front when you want downstream work to bind to a durable snapshot.
uv run latentdna snapshot build anchor60_snapshot \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas \
  --source anchor60

# Materialize one source-backed latent view into canonical matrix form.
uv run latentdna view materialize z20_60 \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas

# Build a deterministic plotting scope shared by downstream artifacts.
# The active promoter study currently exposes the cohort field as `densegen__plan`.
uv run latentdna sample build atlas_sample \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas \
  --view z20_60 \
  --strategy stratified \
  --group-column densegen__plan \
  --n 20000 \
  --seed 17

# Fit a persisted UMAP projection on the sampled scope.
uv run latentdna projection fit z20_60 \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas \
  --sample atlas_sample \
  --run-id umap_z20_60

# Render one review scatter from the persisted projection artifact only.
uv run latentdna plot render anchor_projection_review \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas \
  --kind projection_scatter \
  --projection umap_z20_60 \
  --color-column densegen__plan
```

### Next artifact slice

```bash
# Materialize the paired context view, then compile explicit cross-view support.
uv run latentdna view materialize z20_1k_anchor \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas

# Persist the intersection support that all downstream cross-view work must reuse.
uv run latentdna alignment build anchor_ctx_20b \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas

# Derive the within-model context delta from existing artifacts only.
uv run latentdna view derive delta20 \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas

# Score control distances from explicit landmark declarations.
uv run latentdna distance score primary_landmark_distances \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas \
  --view z20_60 \
  --landmark spy_p \
  --landmark sul_ap

# Turn view/distance artifacts into reusable scalar tables.
uv run latentdna scalar derive delta20_norm \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas

# Build the signed control-margin table from the existing distance artifact.
uv run latentdna scalar derive ethanol_vs_cipro \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas

# Fit a reusable PCA reducer and materialize a reduced view for downstream handoff.
uv run latentdna view reduce delta20 \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas \
  --run-id delta20_pca \
  --dims 32 \
  --reduced-view-id delta20_pc32

# Export the configured low-rank matrix bundle for downstream supervised learning.
uv run latentdna export matrix x1_primary_20b \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas

# Once all four committee projections exist, render the named publication-style atlas recipe.
uv run latentdna plot render atlas_2x2_main \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas
```

### Multiblock export slice

```bash
# Materialize the primary-space reducer output needed for an alignment-basis export.
uv run latentdna view reduce z20_60 \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas \
  --run-id z20_60_pca \
  --dims 32 \
  --reduced-view-id z20_60_pc32

# Combine primary-space PCs, aligned landmark distances, and delta features on the explicit aligned support.
uv run latentdna export matrix x2_primary_20b \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas

# After the 7B views and delta are materialized, export the ablation bundle on the paired 7B alignment.
uv run latentdna export matrix x3_ablation_7b \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas

# Emit the same aligned bundle as a flat table when a downstream consumer prefers tabular handoff.
uv run latentdna export table x2_primary_20b \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas
```

### Structural agreement slice

```bash
# Compile reusable neighborhood structure on an explicit aligned support.
uv run latentdna neighbors fit z20_60_knn \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas \
  --view z20_60 \
  --alignment anchor_ctx_20b \
  --k 15 \
  --backend approximate

# Build the paired challenger neighborhood artifact on the same aligned support.
uv run latentdna neighbors fit z20_1k_anchor_knn \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas \
  --view z20_1k_anchor \
  --alignment anchor_ctx_20b \
  --k 15 \
  --backend approximate

# Add one minimal clustering pass on the same aligned support.
uv run latentdna cluster fit z20_60_kmeans \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas \
  --view z20_60 \
  --alignment anchor_ctx_20b \
  --n-clusters 8

# Mirror the same clustering summary for the challenger view before comparing partitions.
uv run latentdna cluster fit z20_1k_anchor_kmeans \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas \
  --view z20_1k_anchor \
  --alignment anchor_ctx_20b \
  --n-clusters 8

# Compare cross-view local structure without mixing raw coordinates.
uv run latentdna agreement compare agreement_20b_anchor_vs_context \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas \
  --left-neighbors z20_60_knn \
  --right-neighbors z20_1k_anchor_knn \
  --left-clusters z20_60_kmeans \
  --right-clusters z20_1k_anchor_kmeans \
  --landmark spy_p \
  --landmark sul_ap \
  --landmark sox_sp \
  --landmark j23105
```

### Diagnostic plot slice

```bash
# Render control-distance structure directly from the persisted distance table.
uv run latentdna plot render primary_landmark_scatter \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas

# Render one numeric artifact column as a read-only distribution plot.
uv run latentdna plot render spy_distance_distribution \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas \
  --kind distribution \
  --distance primary_landmark_distances \
  --value-column d_spy_p \
  --color-column densegen__plan

# Summarize the compiled agreement metrics without reopening notebooks.
uv run latentdna plot render agreement_20b_anchor_vs_context_summary \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas \
  --kind agreement_summary \
  --agreement agreement_20b_anchor_vs_context
```

### Neighborhood enrichment slice

```bash
# Reuse the persisted neighborhood artifact to score cohort enrichment around control landmarks.
uv run latentdna enrich score control_plan_enrichment \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas \
  --neighbors z20_60_knn \
  --cohort plan \
  --landmark spy_p \
  --landmark sul_ap \
  --landmark sox_sp \
  --landmark j23105

# Render the enrichment table as a read-only heatmap artifact.
uv run latentdna plot render control_plan_heatmap \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas
```

### Thin orchestration slice

```bash
# Validate the checked-in step graph before using it as the automation surface.
uv run latentdna recipe validate control_plan_heatmap_recipe \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas

# Ask whether the user-facing bundle is ready before building anything.
uv run latentdna deliverable status control_neighborhood_enrichment \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas

# Run the declared recipe without hiding the primitive seams.
uv run latentdna deliverable run control_neighborhood_enrichment \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas

# Generate the read-only marimo artifact review app directly from the persisted outputs.
uv run latentdna notebook generate control_plan_review \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas

# Open the generated notebook as an interactive marimo app to inspect declared artifacts
# and any persisted plot artifact under outputs/latentdna/plots inline.
uv run marimo run \
  workspaces/stress_ethanol_cipro_latent_atlas/outputs/latentdna/notebooks/control_plan_review/notebook.py
```

### Operational inventory slice

```bash
# Inspect the persisted artifact inventory and current materialization state.
uv run latentdna inspect artifacts \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas

# Inspect declared views and explicit cross-view support without recomputing anything.
uv run latentdna inspect views \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas

# Inspect one compiled alignment artifact and its persisted support summary.
uv run latentdna inspect alignment anchor_ctx_20b \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas

# Enumerate persisted artifacts in machine-friendly form.
uv run latentdna runs list \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas \
  --json

# Remove an unreferenced artifact explicitly when you want to reclaim workspace state.
uv run latentdna runs prune export_bundle x2_primary_20b \
  --workspace workspaces/stress_ethanol_cipro_latent_atlas
```

### Handoffs

- Use [Infer docs](../../../infer/docs/README.md) when vectors are missing.
- Use [USR README](../../../usr/README.md) when the source dataset itself is missing or needs validation.
- Use [OPAL README](../../../opal/README.md) after `latentdna` has produced a deterministic export bundle.
