# Sigma-35 Ordinal Audit

The audit measures whether each candidate space preserves a declared ordinal
metadata axis. The scalar panel is specific to the DenseGen synthetic Sigma-35
f/e/d/c/b ladder. The row-level ordinal swarm gallery uses the same endpoint
margin grammar for Sigma-35, W Collection core60, and Anderson iGEM core60, but
the W and Anderson numeric standards remain collection-specific scales and
should not be pooled into one promoter-strength score.

For Sigma-35, the scalar audit and the swarm gallery are related, but they are
not the same statistic. Both use the same declared order file, with `f`
strongest and `b` weakest. The swarm displays that ladder as `B=1`, `C=2`,
`D=3`, `E=4`, `F=5`, so a positive row-level Spearman means rows move toward
the `F` endpoint as the class gets stronger. The scalar audit Spearman and
Kendall compare pairwise centroid-distance gaps; the swarm annotations compare
row order against an `F`-vs-`B` endpoint margin.

### sigma35_ordinal_audit | Sigma-35 ordinal audit

#### Plot details

**Data.** Each candidate is evaluated on a stratified sample over the synthetic
promoter population using the workspace Sigma-35 order mapping. Candidate rows
are stored representation summaries from causal Evo 2 passes or the normalized
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

These are not the same Spearman and Kendall values shown on the ordinal swarm
subplot. Here, the question is: are farther-apart declared Sigma-35 classes
farther apart as class centroids in embedding space?

**Decision use.** Ordered Sigma-35 structure adds a within-design signal beyond
coarse cohort separation. A good candidate should preserve the intended ladder
without requiring UMAP layout to make the pattern visible. Future ordinal groups
should answer the same question with the same metric family while keeping their
categorical or numeric scale explicit in the selector.

**Limits.** The current Sigma-35 order file stays exploratory until the exact literature note is checked into the repo. The output measures ordered design structure in pooled embeddings. It does not estimate promoter activity and should not pool Anderson, W Collection, and DenseGen ordinal priors into one score.

### sigma35_margin_ladder_gallery | Ordinal ladder swarm gallery

#### Plot details

**Data.** The gallery uses the same study-facing 7B intermediate candidate
family as the main shortlist. A selector switches among the declared DenseGen
Sigma-35 f/e/d/c/b ladder, W Collection core60 standards, and Anderson iGEM
core60 standards.

**Preprocessing.** Rows are standardized within each view and L2-normalized
before cosine calculations. For the selected ordinal group, LatentDNA forms an
endpoint centroid for the strongest ranked class and an endpoint centroid for
the weakest ranked class. W and Anderson are filtered to core60 reference rows
and are interpreted only within their own collection-specific numeric scales.

**Definition.** The plotted value is the selected group's endpoint margin

$$
m_{\mathrm{ord}}(x) =
\cos(z_x, c_{\mathrm{strong}}) -
\cos(z_x, c_{\mathrm{weak}}),
$$

shown as a swarm-strip ladder. Each point is one row in the selected candidate
view. The black tick marks the class median and the black interval marks the
class interquartile range. Positive values mean a row lies closer to the
strong-end centroid; negative values mean it lies closer to the weak-end
centroid. The annotation reports Spearman rho between row-level ordinal class
order and this high-dimensional margin, plus a simple linear \(R^2\) for those
same two row-level variables.

For the Sigma-35 dropdown selection, the annotation uses the same `f/e/d/c/b`
order as the scalar audit but applies it to row-level endpoint margins. The
question is: do individual rows slide monotonically from `B`-like to `F`-like
along the strong/weak endpoint axis? That is why the swarm statistics can be
larger or smaller than the scalar audit values without contradicting them.

**Decision use.** This is the simplest row-level bridge to the ordinal score.
It shows whether the selected ladder is visible before that order is compressed
into one scalar summary.

**Limits.** The gallery is a derived axis conditioned on study labels and
reference-standard metadata. It is not a promoter-activity plot and should not
replace the high-dimensional scalar summaries.

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
