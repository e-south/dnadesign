# 2026-04-19 LatentDNA Pre-Assay Triage

## Use

This note holds the longer study interpretation that does not belong in the
top-level status and route files.

Keep here:

- commentary on what the current mean-pooled Evo2 views can and cannot support
- external-method comparisons, including Goodfire and EVOLVEpro
- future assay-era extensions and open methodological questions

## Boundary

The checked-in study record still places the live study in
`infer_batch_preparation`.

The pre-assay question is:

> Which mean-pooled Evo2 representation, X, is healthy, preserves trusted
> synthetic-design structure, and remains stable enough under construct context
> to be worth carrying into later supervised modeling?

This is narrower than promoter-function discovery. The current study does not
support claims about promoter fitness, promoter strength, or token-level
mechanism.

## Current View Inventory

The study-facing notebook now surfaces a seven-view 7B inventory:

- `intermediate_embedding_7b_anchor_60bp`
- `pooled_logits_7b_anchor_60bp`
- `intermediate_embedding_7b_full_context_1kb`
- `pooled_logits_7b_full_context_1kb`
- `intermediate_embedding_7b_full_context_anchor_mean`
- `intermediate_embedding_7b_anchor_plus_full_context_concat`
- `intermediate_embedding_7b_anchor_plus_anchor_mean_concat`

Important scope notes:

- `pooled_logits` is the study-facing label for Infer's persisted
  `output_layer_mean` family, which is implemented as mean-pooled per-token
  logits.
- `anchor_60bp` is a study label, not a literal invariant for every row. The
  carried controls are 35 bp, 165 bp, 200 bp, and 220 bp.
- The primary comparison story is still the 7B intermediate family; the 7B
  pooled-logit views stay surfaced as appendix-grade notebook candidates and as
  a log-likelihood hue source for EDA.
- 20B views remain hidden from the notebook and do not participate in the
  current deliverable ladder.

## Shared Geometry Contract

All cosine-based study geometry uses the same preprocessing contract for each
view \(v\):

$$
x'_i = \frac{x_i - \mu_v}{\sigma_v + \epsilon}
$$

and

$$
z_i = \frac{x'_i}{\lVert x'_i \rVert_2 + \epsilon}.
$$

All centroid, cosine-similarity, margin, and cosine-distance calculations are
then computed on \(z_i\), not on the raw stored vectors.

The study compares standardized and row-normalized geometry inside each view,
not raw Evo2 outputs.

Study builders use the collapse-tolerant normalization path. Zero-variance
columns are set to `0.0` after scaling, and zero-norm rows remain `0.0`. This
keeps degenerate spaces finite so they show up in the health gate instead of
failing later cosine calculations.

## What The Current Views Can Test

### 1. Representation health

The gate metric is the effective rank derived from sampled PCA summaries:

$$
r_{\mathrm{eff}} = \exp\left(-\sum_j p_j \log p_j\right), \qquad
p_j = \frac{\lambda_j}{\sum_k \lambda_k}.
$$

This is a capacity diagnostic, not a claim about downstream usefulness. It is
computed from retained explained-variance ratios on sampled reducers, so it is
not the full original spectrum.

### 2. Design-structure preservation

For a design axis \(A\) with cohort \(g\), the study uses normalized cohort
centroids

$$
c_g = \mathrm{normalize}\left(\frac{1}{|g|}\sum_{i \in g} z_i\right)
$$

and the separation ratio

$$
S_A = \frac{B_A}{W_A + \epsilon},
$$

where

$$
B_A = \mathrm{mean}_{g<h}\left(1 - \cos(c_g, c_h)\right)
$$

and

$$
W_A = \mathrm{mean}_g \ \mathrm{mean}_{i \in g}\left(1 - \cos(z_i, c_g)\right).
$$

This asks whether the space preserves trusted study annotations such as design
family, regulator composition, Sigma-35 variant, and spacer length.

### 3. Sigma-35 ordered-axis structure

For the declared Sigma-35 order mapping, the study compares expected rank gaps

$$
\Delta_{\mathrm{rank}}(g,h) = |r_g - r_h|
$$

to observed centroid distances

$$
d_{\mathrm{emb}}(g,h) = 1 - \cos(c_g, c_h).
$$

The audit reports Spearman and Kendall statistics, a balanced Sigma-35
Spearman, a within-family mean Spearman, and a shuffled-label permutation
p-value.

This is the main interpretability signal in the study. It remains
an ordered design-axis audit, not mechanistic interpretability.

### 4. Context robustness

The row-level stability metric is

$$
\mathrm{self\_cos}_i = \cos(z_i^{\mathrm{anchor}}, z_i^{\mathrm{context}}).
$$

For each annotation axis \(A\), the cohort-retention statistic compares the
upper triangles of the anchor and context centroid-distance matrices using
Pearson correlation:

$$
\mathrm{retention}_A =
\rho_{\mathrm{Pearson}}\left(\mathrm{upper}(D_A^{\mathrm{anchor}}),
\mathrm{upper}(D_A^{\mathrm{context}})\right).
$$

This asks whether 1 kb construct context preserves the design geometry already
visible in the anchor view.

### 5. Appendix-only reference alignment

If reference promoters stay in the study, use the appendix-only
background-relative full-context comparison

$$
a_{\mathrm{eth}} =
\cos(c_{\mathrm{eth}}, r_{\mathrm{SpyP}}) -
\cos(c_{\mathrm{bg}}, r_{\mathrm{SpyP}})
$$

and

$$
a_{\mathrm{cipro}} =
\cos(c_{\mathrm{cipro}}, r_{\mathrm{SulA}}) -
\cos(c_{\mathrm{bg}}, r_{\mathrm{SulA}}).
$$

This is weaker than the internal design-structure evidence and stays
appendix-only.

## What The Current Views Cannot Test

The current canonical views do not support:

- assay-era phenotype claims
- token-level mechanistic interpretation
- promoter-function manifold claims
- a primary geodesic story

Without assay-derived phenotype distances, geodesics are mostly a graph choice
layered on top of already weak neighborhood evidence.

## Goodfire And EVOLVEpro Notes

The Goodfire comparison is limited.

- Goodfire SAE-style work is token-level internal-feature analysis. That does
  not port directly because this study persists mean-pooled vectors, not
  tokenwise activations.
- Goodfire phylogeny-style geometry work is methodologically adjacent, but it
  depends on aggregated entity embeddings plus an external biological distance.
  This study has neither yet.
- Goodfire's newer EVEE direction is a warning, not a shortcut: mean pooling
  can blur positional structure.

The EVOLVEpro comparison is also limited.

- EVOLVEpro supports using frozen model features as `X` for a later supervised
  model.
- It does not justify treating model-native geometry as the task-native
  phenotype landscape before assay labels exist.

The present study is feature-space triage, not a promoter-fitness manifold.

## Current Read

Current internal evidence:

- representation health
- design-family structure
- regulator-composition structure
- Sigma-35 structure
- context robustness

Current weak evidence:

- reference-neighbor artifacts
- J23105-relative margin framing
- any kNN-heavy story with little dynamic range

The live study contract therefore uses centroid, separation, and retention
surfaces instead of neighborhood-heavy or reference-heavy narratives.

The current shortlist is still the two 7B intermediate spaces. This is a
pre-assay shortlist, not a final choice.

## Candidate-X Heuristic

Pre-assay, a candidate `X` is only credible if all of the following hold:

1. It is not collapsed.
2. It preserves trusted design annotations.
3. Its useful structure survives added construct context.

So the current study contract is simple:

- health is a gate
- design structure is primary evidence
- Sigma-35 is the main ordered annotation axis
- context robustness tests whether useful geometry survives full-context
  pooling

## Residual Legacy Risk

The live study contract no longer renders the old wildtype, kNN, context-shift,
or tradeoff plots as primary surfaces. Those plots survive only as generated
historical artifacts under `outputs/` and are not part of the current checked-in
workspace contract.

Shared LatentDNA builders still support older benchmark-style surfaces at the
package level, but the active study config now excludes them and the workspace
contract tests assert that exclusion.

## Future Steps

The next study-owned extensions should stay scoped:

1. Surface `anchor_mean` full-context views as appendix-first comparisons if
   the persisted Infer outputs are already present.
2. Keep any geodesic work as a debug or appendix pilot until assay-derived
   phenotype distances exist.
3. When assay data arrives, move from design-geometry audit to
   phenotype-geometry audit:
   - build phenotype vectors
   - compare phenotype distance to direct embedding distance
   - optionally compare phenotype distance to graph geodesics
   - fit small supervised top-layer regressors with hard holdouts

## Placement

Keep this kind of synthesis here.

Keep these top-level study files concise and record-backed:

- `docs/studies/stress_ethanol_cipro_growth/status.md`
- `docs/studies/stress_ethanol_cipro_growth/routes.md`
- `docs/studies/stress_ethanol_cipro_growth/pipeline.yaml`

They should stay scoped and cheap to refresh.
