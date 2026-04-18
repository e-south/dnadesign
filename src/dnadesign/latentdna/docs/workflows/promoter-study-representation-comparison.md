# Promoter-Study Representation Comparison

**Type:** workflow
**Plane:** data-plane
**Surface role:** downstream-analysis
**Owner-boundary:** latentdna
**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-17
**Registry-id:** latentdna.promoter-study.representation-comparison
**Entry artifact:** promoter/stress_ethanol_cipro_anchor_set and promoter/stress_ethanol_cipro_construct_contexts
**Exit artifact:** published LatentDNA workspace snapshot plus sanctioned comparison deliverables and the plot-review notebook

LatentDNA is study-neutral at the primitive level. The promoter study supplies workspace binding, dataset identity, required references, and the review order through `docs/studies/stress_ethanol_cipro_growth/latentdna_binding.yaml` and the published workspace snapshot.

### Primary review path

1. `dataset_overview`
2. `reference_margin_analysis`
3. `context_geometry_audit`
4. `representation_comparison`
5. `representation_health_diagnostic`

### Appendix surfaces

- `reference_margin_gallery_synthetic_centroids`
- `appendix_umap_gallery`

### Notebook role

- `latent_geometry_browser` stays a single notebook artifact.
- `Plots` is the primary surface for reviewing persisted plots in scientific order.
- `Geometry audit` and `Comparison audit` remain available as secondary audit tabs.
- Plot-level interpretation belongs in the study deliverable markdown, not in this workflow doc.

### Canonical geometry inventory

- `intermediate_embedding_20b_anchor_60bp`
- `intermediate_embedding_20b_full_context_1kb`
- `intermediate_embedding_7b_anchor_60bp`
- `intermediate_embedding_7b_full_context_1kb`
- `pooled_logits_20b_anchor_60bp`
- `pooled_logits_20b_full_context_1kb`
- `pooled_logits_7b_anchor_60bp`
- `pooled_logits_7b_full_context_1kb`

### Operator path

```bash
# Publish the study-facing snapshot.
uv run latentdna workspace snapshot --workspace stress_ethanol_cipro_growth --json
# Verify the workspace contract after config or docs changes.
uv run latentdna validate workspace --workspace stress_ethanol_cipro_growth --deep
# Check whether reference-margin deliverables are fresh enough to review.
uv run latentdna deliverable status reference_margin_analysis --workspace stress_ethanol_cipro_growth
# Check whether the context-stability deliverables are fresh enough to review.
uv run latentdna deliverable status context_geometry_audit --workspace stress_ethanol_cipro_growth
# Rebuild the plot-review notebook after persisted artifacts refresh.
uv run latentdna notebook generate latent_geometry_browser --workspace stress_ethanol_cipro_growth
```

### Guardrails

- Do not choose representations by UMAP aesthetics.
- Do not claim anchor-local mechanism from pooled full-context vectors.
- Do not treat the notebook browser as the authoritative study-status surface.
- Use the workspace snapshot for downstream LatentDNA posture after reading the checked-in study record.
