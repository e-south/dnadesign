# Promoter-Study Representation Comparison

**Type:** workflow
**Plane:** data-plane
**Surface role:** downstream-analysis
**Owner-boundary:** latentdna
**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-19
**Registry-id:** latentdna.promoter-study.representation-comparison
**Entry artifact:** promoter/stress_ethanol_cipro_anchor_set and promoter/stress_ethanol_cipro_construct_contexts
**Exit artifact:** published LatentDNA workspace snapshot plus sanctioned comparison deliverables and the `latent_geometry_browser` notebook

The promoter study binds LatentDNA to two published datasets and eight canonical mean-pooled Evo2 views. The active contract is pre-assay representation triage: choose a plausible mean-pooled Evo2 feature space \(X\) for later supervised modeling.

### Gate

1. `representation_health_summary`

### Primary review path

1. `dataset_overview`
2. `design_structure_summary`
3. `sigma35_ordinal_audit`
4. `context_robustness_summary`

### Appendix surfaces

- `design_centroid_margin_gallery`
- `reference_alignment_summary`
- `representation_scree_diagnostic`
- `appendix_umap_gallery`

### Notebook role

- `latent_geometry_browser` stays a single notebook artifact.
- `Plots` is the primary surface for reviewing persisted plots in scientific order.
- Plot-level interpretation belongs in the study deliverable markdown, not in this workflow doc.

### Shared Geometry Contract

Across the study-facing geometry surfaces, each view is first standardized by
view and then row-normalized:

$$
x'_i = \frac{x_i - \mu_v}{\sigma_v + \epsilon},
\qquad
z_i = \frac{x'_i}{\lVert x'_i \rVert_2 + \epsilon}.
$$

Cosine, centroid, margin, and cosine-distance calculations operate on
\(z_i\).

`appendix_umap_gallery` does not use this contract. Its persisted UMAP fits are
built from the raw stored view matrices unless a projection manifest says
otherwise.

### Deeper study notes

Keep longer interpretation in
`src/dnadesign/studies/stress_ethanol_cipro_growth/notes/`.

Use the notes for:

- method commentary about the current LatentDNA views
- external analogies such as Goodfire or EVOLVEpro comparisons
- future assay-era extensions and open methodological questions

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
# Check whether the representation-health gate is fresh enough to review.
uv run latentdna deliverable status representation_health_summary --workspace stress_ethanol_cipro_growth
# Check whether the context-robustness summary is fresh enough to review.
uv run latentdna deliverable status context_robustness_summary --workspace stress_ethanol_cipro_growth
# Rebuild the `latent_geometry_browser` notebook after persisted artifacts refresh.
uv run latentdna notebook generate latent_geometry_browser --workspace stress_ethanol_cipro_growth
```

### Guardrails

- Do not choose representations by UMAP aesthetics.
- These plots do not estimate promoter activity.
- Direct cosine and centroid summaries are the active analysis path.
- Leave geodesic pilots in study notes until phenotype distances exist.
- Use the checked-in study record for study phase and the workspace snapshot for the latest LatentDNA outputs.
