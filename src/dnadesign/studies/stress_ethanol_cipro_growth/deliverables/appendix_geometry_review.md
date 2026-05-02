# Appendix Geometry Review

These appendix surfaces provide orientation and diagnostics. Candidate ranking stays on the primary review path.

### design_centroid_margin_gallery | Design-centroid margin gallery

#### Plot details

**Data.** Each point is one promoter row in a candidate representation
space. The gallery uses study-internal design cohorts rather than external
reference promoters.

**Preprocessing.** Margins use view-level standardization followed by row L2 normalization and leakage-safe leave-one-out cohort centroids.

**Definition.** The axes are

$$
m_{\mathrm{eth}}(x) = \cos(z_x, c_{\mathrm{eth}}) - \cos(z_x, c_{\mathrm{bg}})
$$

and

$$
m_{\mathrm{cipro}}(x) = \cos(z_x, c_{\mathrm{cipro}}) - \cos(z_x, c_{\mathrm{bg}}).
$$

**Decision use.** The scatter gives a quick orientation view of internal design geometry.

**Limits.** The scatter compresses high-dimensional geometry to two margins. Use the summary metrics for ranking.

### reference_alignment_summary | Reference collapse summary

#### Plot details

**Data.** Anchor and forward 1 kb anchor-mean candidate samples that include
native references, derived core60 rows, SFXI controls, Anderson iGEM standards,
and the T7/W collection when those labels are present in the materialized rows.

**Preprocessing.** Uses the same view-level standardization and row
normalization contract as the other cosine-based plots. Collection summaries
are resolved from workspace `reference_sets`, so missing or incomplete
collections remain explicit status rows; strength scales are not pooled into
one biological scale.

**Definition.** The legacy stress-reference panels still report
background-relative alignment:

$$
a_{\mathrm{eth}} = \cos(c_{\mathrm{eth}}, r_{\mathrm{SpyP}}) - \cos(c_{\mathrm{bg}}, r_{\mathrm{SpyP}})
$$

and

$$
a_{\mathrm{cipro}} = \cos(c_{\mathrm{cipro}}, r_{\mathrm{SulA}}) - \cos(c_{\mathrm{bg}}, r_{\mathrm{SulA}}).
$$

The collection-collapse panels report group size, median pairwise cosine
distance, and pairwise cosine-distance IQR within each configured reference
set.

**Decision use.** The panel keeps reference landmarks and collapse diagnostics
visible as appendix evidence while candidate selection remains on the primary
ladder.

**Limits.** Reference alignment stays well below assay-era evidence. Poor
alignment does not automatically invalidate a candidate X. Small groups,
mixed collection scales, and context dilution can make group-level distances
look more collapsed or separated than the underlying biology supports.

### representation_scree_diagnostic | PCA variance decay

#### Plot details

**Data.** Sample-based PCA reducer summaries for each candidate representation.

**Preprocessing.** Uses the persisted sampled reducer summaries directly.

**Definition.** Each panel shows retained variance ratios and cumulative retained variance over the stored principal components.

**Decision use.** The scree panels expose the representation-health gate directly.

**Limits.** Scree shape alone does not say whether the retained directions are useful for later supervised modeling.
