# Sigma-35 Ordinal Audit

The audit measures whether each candidate space preserves a declared ordinal
metadata axis. The current rendered group is the DenseGen synthetic Sigma-35
f/e/d/c/b ladder. Anderson iGEM strength and W Collection strength are numeric
standard annotations with their own collection scales, so they should be audited
with collection-specific statistics rather than pooled into this Sigma-35
ladder.

### sigma35_ordinal_audit | Sigma-35 ordinal audit

#### Plot details

**Data.** Each candidate is evaluated on a stratified sample over the synthetic
promoter population using the workspace Sigma-35 order mapping. Candidate rows
are stored representation summaries from causal Evo2 passes or the normalized
forward/RC concat view. This group is intentionally separate from
reference-standard strength scales.

**Preprocessing.** All centroid distances use view-level standardization followed by row L2 normalization. Study builders use the collapse-tolerant path: zero-variance columns are set to `0.0` after scaling, and zero-norm rows remain `0.0`.

**Definition.** For the declared Sigma-35 order, the audit compares expected
rank gaps

$$
\Delta_{\mathrm{rank}}(g,h) = |r_g - r_h|
$$

to observed centroid distances

$$
d_{\mathrm{emb}}(g,h) = 1 - \cos(c_g, c_h).
$$

It reports Spearman and Kendall correlations between those two vectors, plus a
balanced Sigma-35 Spearman, a within-family mean Spearman, a within-regulator
mean Spearman, and a shuffled-label permutation p-value. Confidence intervals
are reported for the Spearman-based summary rows.

**Decision use.** Ordered Sigma-35 structure adds a within-design signal beyond
coarse cohort separation. A good candidate should preserve the intended ladder
without requiring UMAP layout to make the pattern visible. Future ordinal groups
should answer the same question with the same metric family while keeping their
categorical or numeric scale explicit in the selector.

**Limits.** The current Sigma-35 order file stays exploratory until the exact literature note is checked into the repo. The output measures ordered design structure in pooled embeddings. It does not estimate promoter activity and should not pool Anderson, W Collection, and DenseGen ordinal priors into one score.

### sigma35_margin_ladder_gallery | Sigma-35 margin ladder gallery

#### Plot details

**Data.** The gallery uses the same study-facing 7B intermediate candidate family as the main shortlist and renders the declared DenseGen Sigma-35 f/e/d/c/b audit subset in each candidate view.

**Preprocessing.** All margins are computed after view-level standardization and row L2 normalization. Leave-one-out handling is used when a row belongs to one of the scored Sigma-35 cohorts.

**Definition.** The plotted value is the Sigma-35 F-vs-B centroid margin

$$
m_{\sigma35}(x) = \cos(z_x, c_f) - \cos(z_x, c_b),
$$

shown as a violin-and-box ladder across the declared f/e/d/c/b Sigma-35 variants. Reference-derived or unranked annotated Sigma-35 sequences are intentionally not given extra x-axis categories. Positive values mean a row sits closer to the `f` centroid than the `b` centroid; more negative values mean the opposite.

**Decision use.** This is the simplest row-level bridge to the Sigma-35 ordinal Spearman. It shows whether the intended `f > e > d > c > b` ladder is visible before that order is compressed into one scalar score.

**Limits.** The gallery is a single derived axis conditioned on study labels. It helps justify the ordinal audit, but it is not a promoter-activity plot and should not replace the scalar summary.

### sigma35_stress_margin_gallery | Sigma-35 vs stress-margin gallery

#### Plot details

**Data.** The gallery uses the same study-facing 7B intermediate candidate family as the main shortlist and shows the full shared promoter population in each candidate view.

**Preprocessing.** All margins are computed after view-level standardization and row L2 normalization. Leave-one-out handling is used when a row belongs to one of the scored cohorts.

**Definition.** The x axis is the Sigma-35 F-vs-B centroid margin,

$$
m_{\sigma35}(x) = \cos(z_x, c_f) - \cos(z_x, c_b),
$$

so positive x values mean a row is closer to the `f` centroid than the `b` centroid.

and the y axis is the best stress-family margin,

$$
m_{\mathrm{stress}}(x) = \max\left(m_{\mathrm{eth}}(x), m_{\mathrm{cipro}}(x)\right),
$$

with

$$
m_{\mathrm{eth}}(x) = \cos(z_x, c_{\mathrm{eth}}) - \cos(z_x, c_{\mathrm{bg}})
$$

and

$$
m_{\mathrm{cipro}}(x) = \cos(z_x, c_{\mathrm{cipro}}) - \cos(z_x, c_{\mathrm{bg}}).
$$

Positive $m_{\mathrm{stress}}(x)$ values mean the row is closer to at least one stress-family centroid than to background.

**Decision use.** This is the deterministic row-level companion to the Sigma-35 audit. It shows whether a candidate keeps an ordered Sigma-35 axis visible while still separating the study's stress-family directions without relying on UMAP layout.

**Limits.** The gallery is still annotation-conditioned and two-dimensional. It is useful for communication and orientation, but it should not replace the scalar ordinal audit or be interpreted as promoter activity.

### sigma35_centroid_distance_gallery | Sigma-35 centroid-distance gallery

#### Plot details

**Data.** The gallery uses the same declared f/e/d/c/b Sigma-35 ladder as the ordinal audit and restricts the visual to the study-facing 7B intermediate candidates.

**Preprocessing.** Variant centroids are computed after view-level standardization and row L2 normalization on the full materialized candidate view for this companion heatmap. The rendered matrix keeps only the configured f/e/d/c/b axis even when source/reference rows carry additional annotated Sigma-35 sequences. Distances are cosine distances between those normalized centroids.

**Definition.** For each variant \(g\),

$$
c_g = \mathrm{normalize}\left(\frac{1}{|g|}\sum_{i \in g} z_i\right),
$$

and each heatmap entry is

$$
d_{\mathrm{emb}}(g,h) = 1 - \cos(c_g, c_h).
$$

Smaller values mean two Sigma-35 variant centroids stay close in the representation; larger values mean they are farther apart.

**Decision use.** This gallery is appendix-only support for the scalar ordinal audit. It shows whether nearby variants in the declared ladder also remain nearby in centroid space.

**Limits.** Pairwise centroid distances still compress higher-dimensional geometry. The gallery is not a promoter-activity plot and should not replace the scalar audit when ranking candidates.
