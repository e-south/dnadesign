# Stress Ethanol/Cipro Representation Comparison

**Type:** workflow
**Plane:** data-plane
**Surface role:** downstream-analysis
**Owner-boundary:** latentdna
**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-08
**Registry-id:** latentdna.stress-ethanol-cipro-growth.representation-comparison
**Entry artifact:** usr_prom_eth_cip_anchor, construct_prom_eth_cip_context, usr_promoter_references, construct_prom_eth_cip_reference_core60, and construct_prom_eth_cip_reference_contexts
**Exit artifact:** published LatentDNA workspace snapshot plus sanctioned comparison deliverables and the `latent_geometry_browser` notebook

The stress_ethanol_cipro_growth study binds LatentDNA to row-level USR metadata sources plus
canonical Infer feature sidecars. The row sources remain useful for cohort,
landmark, and dataset-overview plots. Embedding-bearing views must come from
`_derived/infer/feature_aliases.parquet` joined to `feature_vectors.parquet`
and the dataset-local sequence-view sidecars; LatentDNA no longer depends on
USR row-overlay embedding columns for the active 7B study surfaces.
Sigma-35 metadata is a study-configured annotation derivation, not a LatentDNA
package default: this workspace explicitly derives `sig35_variant` from
DenseGen plan tokens, DenseGen fixed-element details, USR `seq_annot` `-35`
features, or Construct retained-feature bounds. Annotated unranked hexamers
are kept in source inventory and eligible plots; only ordinal-rank statistics
restrict themselves to the explicit b-f order file. The package-level scalar
primitives are the generic `ordinal_axis_audit` and `axis_centroid_distance`;
this workspace config maps those axis builders onto Sigma-35 metric names and
heatmap labels. The Sigma-35 metric ids are declared under workspace
`metric_definitions`; the global LatentDNA metric registry does not ship those
study-specific names or mathematical definitions.
Sigma-35 display labels, color order, the ordinal subset, and the
reference/other noncanonical bucket are declared under `metadata.axes` in the
stress workspace config. The notebook and renderer consume those resolved axis
styles from `controls.json` or the workspace config; they do not contain
Sigma-35 column-name branches.

The active contract is still pre-assay representation triage: choose a plausible
mean-pooled Evo2 feature space \(X\) for later supervised modeling. The current
7B sidecar-backed decision geometry is intentionally small: anchor-source
`seq_mean`, forward 1 kb context `anchor_mean`, and one controlled equal-block
bidirectional forward/RC context `anchor_mean` concat. Forward and
reverse-complement full-context `seq_mean` views remain diagnostics unless they
beat the predeclared candidate-X checks. Reference core60 and reference-context
views are hidden audit geometries for reference-normalization questions.
Output-layer mean views store mean-pooled per-token output-logits summaries,
and log-likelihood scalar diagnostics are collected for QC. Neither family is
current decision geometry.

### Mean-Pooling Semantics

Infer writes pooled Evo2 vectors before LatentDNA reads them. For a sequence
`x_1, ..., x_T`, the per-token layer output at position `t` is treated as a
causal, prefix-conditioned state:

$$
h_t^{(\ell)} = f_\theta^{(\ell)}(x_{\le t}).
$$

For a selected span `I = [a, b)`, the stored span vector is

$$
z_I^{(\ell)} = |I|^{-1}\sum_{t \in I} h_t^{(\ell)}.
$$

This is a prefix-conditioned causal mean-pooled span embedding. The mean is
over sequence positions, not over embedding dimensions. Later positions in the
span have seen earlier positions in that same emitted sequence, but earlier
positions have not seen downstream bases. Therefore a forward 1 kb
`anchor_mean` should be described as "the anchor-span mean pooled from a full
forward 1 kb causal pass," not as a native bidirectional Evo2 hidden state.

The reverse-complement context views are separate full-sequence Evo2 passes over
reverse-complement emitted sequences with Construct-provided emitted-orientation
anchor bounds. The controlled bidirectional candidate is a LatentDNA-derived
external summary:

$$
X_{\mathrm{bidir}} =
[\mathrm{L2}(\mathrm{Std}(Z_{\mathrm{fwd}}));
\mathrm{L2}(\mathrm{Std}(Z_{\mathrm{rc}}))]
$$

where `Std` is fit over the materialized block rows before row L2
normalization. It is useful because it combines forward and reverse-complement
causal summaries with equal block weight. In study prose, "bidirectional 1 kb
view" is acceptable shorthand only when expanded as "forward plus
reverse-complement causal 1 kb anchor-span summaries." It is not a native bidirectional Evo2 hidden state.

### View-Language Glossary

Use these descriptions in study prose, plot accordions, and figure alt text:

- **Anchor-source `seq_mean`:** Evo2 is run on the emitted source insert and
  Infer averages the token-position vectors across that insert. This is the
  conservative DenseGen-plan baseline. The merged source is mostly 60 bp, but
  native references and controls can have other lengths.
- **Full-context `seq_mean`:** Evo2 is run on the full emitted 1 kb construct
  context and Infer averages all 1 kb token positions. This summarizes the
  whole construct window and can dilute anchor-local promoter grammar.
- **Full-context `anchor_mean`:** Evo2 is run on the full emitted 1 kb context,
  but Infer averages only the Construct-provided anchor span. This asks what
  the promoter insert looks like inside vector context under causal
  left-to-right token states.
- **Reverse-complement context views:** Construct emits a matched
  reverse-complement 1 kb sequence with reverse-complement-orientation anchor
  bounds. Infer runs Evo2 on that emitted sequence as a separate causal pass.
- **Forward/RC `anchor_mean` concat:** LatentDNA standardizes and
  row-L2-normalizes the forward and reverse-complement `anchor_mean` matrices
  separately, aligns rows by the configured join key, and concatenates the two
  blocks along the feature axis. This is an external equal-block
  two-orientation row summary, not a native bidirectional Evo2 hidden state.
- **Output-layer mean:** Infer applies the same pooling scopes to per-token
  logits. These views are diagnostic QC surfaces, not the current preferred
  candidate `X`.

Avoid saying that a pooled view "sees the whole interval" unless the sentence
also states the causal caveat. A span mean averages token states that have
different prefix lengths; later positions have seen earlier bases, but earlier
positions have not seen later bases.

### Gate

1. `representation_health_summary`

### Primary review path

1. `dataset_overview`
2. `design_structure_summary`
3. `sigma35_ordinal_audit`
4. `context_robustness_summary`
5. `candidate_decision_frontier`
6. `candidate_x_selection_scorecard`

### Companion visuals

- `balanced_design_family_margin_gallery`
- `sigma35_margin_ladder_gallery`
- `sigma35_stress_margin_gallery`
- `context_pair_summary`
- `reference_to_plan_centroid_heatmap`
- `reference_standard_strength_audit`

### Appendix surfaces

- `sigma35_centroid_distance_gallery`
- `design_centroid_margin_gallery`
- `reference_alignment_summary`
- `representation_scree_diagnostic`
- `appendix_umap_gallery`

Named reference collections in `reference_alignment_summary` are selected
through workspace `reference_sets`. Native MG1655, Anderson iGEM, W collection,
spyP/sulAp, SFXI, or future non-promoter landmarks therefore share the same
selector, label, overlay, and missing-status contract.

### Notebook role

- `latent_geometry_browser` stays a single notebook artifact.
- The generated Marimo app has one dropdown-driven surface, not separate review
  and geometry tabs.
- Grid mode is the default for multi-view candidate sets, while single-view
  geometry remains a first-class control.
- Grid presets show only views with persisted projection artifacts by default.
  Planned or materialized views without UMAP/projection outputs remain visible
  in candidate inventory, health, and alignment summaries rather than appearing
  as blank projection-browser panels.
- Plot-level and view-level accordions render in the selected surface. Their
  interpretation belongs in the study deliverable markdown, not in this
  workflow doc.

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
Representation-health pairwise distance summaries use all row pairs when the
candidate sample has at most `pairwise_max_rows` rows, currently defaulting to
4096. Larger samples use a deterministic row sample keyed by `pairwise_seed`
before computing all pairs in that sampled set, and the scalar output records
the evaluated row count, pair count, seed, and method.

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
`src/dnadesign/studies/units/stress_ethanol_cipro_growth/notes/`.

Use the notes for:

- method commentary about the current LatentDNA views
- external analogies such as Goodfire or EVOLVEpro comparisons
- future assay-era extensions and open methodological questions

### Surfaced Notebook Inventory

- `intermediate_embedding_7b_anchor_60bp`
- `intermediate_embedding_7b_full_context_1kb`
- `intermediate_embedding_7b_full_context_anchor_mean`
- `intermediate_embedding_7b_context_anchor_mean_bidir_concat`
- `intermediate_embedding_7b_reverse_complement_context_1kb`
- `intermediate_embedding_7b_reverse_complement_context_anchor_mean`

The candidate-X selection set is narrower than the full surfaced notebook
inventory:

- `intermediate_embedding_7b_anchor_60bp` is the conservative DenseGen-plan
  baseline.
- `intermediate_embedding_7b_full_context_anchor_mean` is the strength-standard
  interpretation lens.
- `intermediate_embedding_7b_context_anchor_mean_bidir_concat` is the current
  working pre-assay `X`.

The controlled concat surface uses equal-block forward/RC anchor-mean
normalization. Raw concat remains out of scope. Stress-study output-layer mean
vectors and log-likelihood total or mean-per-token values are diagnostic QC
surfaces. 20B surfaces are not active in this workspace contract.

### RegulonDB note

The separate `regulondb_native_promoter_panel` workspace uses the same generic
LatentDNA machinery. Its core60 source is a TSS-upstream `[0,60)` analysis
window from native 81 bp records, not a -10/-35 centered promoter box window.
RegulonDB sigma-factor, confidence, and completeness labels are native row
metadata or explicit lookup derivations; they are not aliases for
`sig35_variant`. Output-layer mean views are materialized representation
summaries and stay in candidate inventory, representation-health metrics, and
native/core60 alignment summaries. They are not default projection-browser
panels unless a real output-layer projection artifact is fit and validated.

The stress-study native TF-axis audit is narrower: RegulonDB native core60 rows
are planned as an append-only cohort in the existing `usr_prom_eth_cip_anchor`
and `construct_prom_eth_cip_context` handoff. The audit reuses the same
forward/RC context-anchor bidirectional view as candidate-X triage, filters
rows by `derived__parent_dataset`, and joins BaeR/CpxR/LexA flags from the
RegulonDB regulatory-interaction sidecar. It is not a separate Construct
context project and is not an OPAL input.

### Sequence-View Plot Contract

- `anchor_mean` plots use full emitted 1 kb context sequences and pool over
  Construct-provided emitted-orientation anchor bounds. They do not truncate the
  sequence before Infer or plotting, and they do not give every token in the
  span downstream access under Evo2's causal token-state semantics.
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
- Existing view artifacts built from USR row-overlay embedding columns are stale under
  this contract. Deep validation reports source-contract drift; rerun view
  materialization from sidecars before using refreshed plots as current evidence.

### Operator path

Set `MPLCONFIGDIR=/tmp/dnadesign_mpl` on hosts where Matplotlib cannot write
its default cache directory. Regenerate deliverables only when the status or
snapshot command reports stale artifacts.

```bash
# Publish the study-facing snapshot.
MPLCONFIGDIR=/tmp/dnadesign_mpl uv run latentdna workspace snapshot --workspace stress_ethanol_cipro_growth --json
# Verify the workspace contract after config or docs changes.
MPLCONFIGDIR=/tmp/dnadesign_mpl uv run latentdna validate workspace --workspace stress_ethanol_cipro_growth --deep
# Inspect the canonical sidecar-backed active sources.
uv run latentdna inspect source anchor_7b_seq_mean_features --workspace stress_ethanol_cipro_growth --json
# Inspect the paired forward-context anchor-mean sidecar source.
uv run latentdna inspect source full_context_7b_forward_anchor_mean_features --workspace stress_ethanol_cipro_growth --json
# Inspect the planned reference core60 sidecar source without requiring completed vectors.
uv run latentdna inspect source reference_core60_7b_core60_mean_features --workspace stress_ethanol_cipro_growth --json
# Check whether the representation-health gate is fresh enough to review.
MPLCONFIGDIR=/tmp/dnadesign_mpl uv run latentdna deliverable status representation_health_summary --workspace stress_ethanol_cipro_growth
# Check whether the context-robustness summary is fresh enough to review.
MPLCONFIGDIR=/tmp/dnadesign_mpl uv run latentdna deliverable status context_robustness_summary --workspace stress_ethanol_cipro_growth
# Refresh the default review deliverable if freshness drift is reported.
MPLCONFIGDIR=/tmp/dnadesign_mpl uv run latentdna deliverable run representation_health_summary --workspace stress_ethanol_cipro_growth
# Rebuild the `latent_geometry_browser` notebook after persisted artifacts refresh.
MPLCONFIGDIR=/tmp/dnadesign_mpl uv run latentdna notebook generate latent_geometry_browser --workspace stress_ethanol_cipro_growth
```

### Guardrails

- Do not choose representations by UMAP aesthetics.
- These plots do not estimate promoter activity.
- Direct cosine and centroid summaries are the active analysis path.
- Leave geodesic pilots in study notes until phenotype distances exist.
- Use the checked-in study record for study phase and the workspace snapshot for the latest LatentDNA outputs.
