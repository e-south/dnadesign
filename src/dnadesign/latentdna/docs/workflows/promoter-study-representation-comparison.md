# Promoter-Study Representation Comparison

**Type:** workflow
**Plane:** data-plane
**Surface role:** downstream-analysis
**Owner-boundary:** latentdna
**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-28
**Registry-id:** latentdna.promoter-study.representation-comparison
**Entry artifact:** usr_prom_eth_cip_anchor and construct_prom_eth_cip_context
**Exit artifact:** published LatentDNA workspace snapshot plus sanctioned comparison deliverables and the `latent_geometry_browser` notebook

The promoter study now binds LatentDNA to row-level USR metadata sources plus
canonical Infer feature sidecars. The row sources remain useful for cohort,
landmark, and dataset-overview plots. Embedding-bearing views must come from
`_derived/infer/feature_aliases.parquet` joined to `feature_vectors.parquet`
and the dataset-local sequence-view sidecars; LatentDNA no longer depends on
legacy row-overlay embedding columns for the active 7B study surfaces.
Sigma-35 metadata is source-backed, not ladder-hardcoded: builders derive
`sig35_variant` from DenseGen plan tokens, DenseGen fixed-element details, USR
`seq_annot` `-35` features, or Construct retained-feature bounds. Annotated
unranked hexamers are kept in source inventory and compatible plots; only
ordinal-rank statistics restrict themselves to the explicit b-f order file.

The active contract is still pre-assay representation triage: choose a plausible
mean-pooled Evo2 feature space \(X\) for later supervised modeling. Available
sidecar-backed geometry is narrower than the future target set: the current
usable features are 7B construct-insert `seq_mean` anchors and 7B forward
realized-context `anchor_mean` features. Forward context `seq_mean`,
reverse-complement context features, reference `analysis_window` features,
mean-pooled output-layer logits, and log-likelihood scalar diagnostics are not
treated as current decision geometry until their canonical vector or scalar
sidecars are present.

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
- `intermediate_embedding_7b_full_context_anchor_mean`

Planned sidecar-backed surfaces are declared for forward full-context sequence
mean, reverse-complement full-context sequence mean, and reverse-complement
context anchor mean, but they currently select zero feature aliases. Output-layer
mean vectors are declared as planned vector views. Log-likelihood total and
mean-per-token values are tracked by Infer scalar sidecars and remain
diagnostic/QC surfaces, not active LatentDNA geometry defaults. 20B and concat
surfaces are not active in this workspace contract.

### Sequence-View Plot Contract

- `anchor_mean` plots use full emitted 1 kb context sequences and pool over
  Construct-provided emitted-orientation anchor bounds. They do not truncate the
  sequence before Infer or plotting.
- Reverse-complement context plots require materialized reverse-complement
  context feature aliases. LatentDNA must not reverse-complement sequences or
  synthesize missing products.
- Feature-backed plots include only rows with canonical Infer feature aliases.
  If an annotated SFXI, reference, or analysis-window row is absent from an
  embedding plot, that is missing Infer feature coverage, not a Sigma-35
  category filter.
- Reference-normalization plots will use explicit `analysis_window` and
  `realized_context` sequence-view features once `infer_prom_eth_cip_reference_views_7b`
  exists. Native exact-60 source rows are not relabeled as analysis windows.
- Existing view artifacts built from legacy row-overlay columns are stale under
  this contract. Deep validation reports source-contract drift; rerun view
  materialization from sidecars before using refreshed plots as current evidence.

### Operator path

```bash
# Publish the study-facing snapshot.
uv run latentdna workspace snapshot --workspace stress_ethanol_cipro_growth --json
# Verify the workspace contract after config or docs changes.
uv run latentdna validate workspace --workspace stress_ethanol_cipro_growth --deep
# Inspect the canonical sidecar-backed active sources.
uv run latentdna inspect source anchor_7b_seq_mean_features --workspace stress_ethanol_cipro_growth --json
# Inspect the paired forward-context anchor-mean sidecar source.
uv run latentdna inspect source full_context_7b_forward_anchor_mean_features --workspace stress_ethanol_cipro_growth --json
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
