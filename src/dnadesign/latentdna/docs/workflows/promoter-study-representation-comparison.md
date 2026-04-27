# Promoter-Study Representation Comparison

**Type:** workflow
**Plane:** data-plane
**Surface role:** downstream-analysis
**Owner-boundary:** latentdna
**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-22
**Registry-id:** latentdna.promoter-study.representation-comparison
**Entry artifact:** usr_prom_eth_cip_anchor and construct_prom_eth_cip_context
**Exit artifact:** published LatentDNA workspace snapshot plus sanctioned comparison deliverables and the `latent_geometry_browser` notebook

The promoter study binds LatentDNA to two published datasets and a 7B-first notebook candidate inventory. The live notebook surface exposes the real seven-geometry 7B-first browser, while 20B views remain secondary and materializable for debug-only review. This browser-default posture matches the study record's current infer-runtime preference. The active contract is pre-assay representation triage: choose a plausible mean-pooled Evo2 feature space \(X\) for later supervised modeling.

### Gate

1. `representation_health_summary`

### Primary review path

1. `dataset_overview`
2. `design_structure_summary`
3. `sigma35_ordinal_audit`
4. `context_robustness_summary`
5. `candidate_decision_frontier`

### Companion visuals

- `balanced_design_family_margin_gallery`
- `sigma35_margin_ladder_gallery`
- `sigma35_stress_margin_gallery`
- `context_pair_summary`

### Appendix surfaces

- `sigma35_centroid_distance_gallery`
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

Study builders use the collapse-tolerant normalization path from
`standardize_and_l2_normalize(..., zero_variance_policy="drop_or_zero",
zero_row_policy="zero")`. Zero-variance columns are set to `0.0` after
scaling, and zero-norm rows remain zero vectors so degenerate spaces stay
finite and surface in the health gate instead of failing later cosine plots.

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

### Surfaced Notebook Inventory

- `intermediate_embedding_7b_anchor_60bp`
- `pooled_logits_7b_anchor_60bp`
- `intermediate_embedding_7b_full_context_1kb`
- `pooled_logits_7b_full_context_1kb`
- `intermediate_embedding_7b_full_context_anchor_mean`
- `intermediate_embedding_7b_anchor_plus_full_context_concat`
- `intermediate_embedding_7b_anchor_plus_anchor_mean_concat`

20B views remain materializable in the workspace but are hidden from the study notebook and deliverable ladder.

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
# Refresh the default review deliverable if freshness drift is reported.
uv run latentdna deliverable run representation_health_summary --workspace stress_ethanol_cipro_growth
# Rebuild the `latent_geometry_browser` notebook after persisted artifacts refresh.
uv run latentdna notebook generate latent_geometry_browser --workspace stress_ethanol_cipro_growth
```

### Guardrails

- Do not choose representations by UMAP aesthetics.
- These plots do not estimate promoter activity.
- Direct cosine and centroid summaries are the active analysis path.
- Leave geodesic pilots in study notes until phenotype distances exist.
- Use the checked-in study record for study phase and the workspace snapshot for the latest LatentDNA outputs.
