# Appendix Geometry Audit

These appendix surfaces provide orientation and diagnostics. Candidate ranking stays on the primary review path.

### design_centroid_margin_gallery | Design-centroid margin gallery

#### Plot details

**Data.** Each point is one sampled promoter row in a candidate representation
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

### reference_alignment_summary | Reference-alignment summary

#### Plot details

**Data.** Full-context candidate samples that include the carried
stress-reference promoters alongside the synthetic population.

**Preprocessing.** Uses the same view-level standardization and row normalization contract as the other cosine-based plots.

**Definition.** The panel reports background-relative reference alignment:

$$
a_{\mathrm{eth}} = \cos(c_{\mathrm{eth}}, r_{\mathrm{SpyP}}) - \cos(c_{\mathrm{bg}}, r_{\mathrm{SpyP}})
$$

and

$$
a_{\mathrm{cipro}} = \cos(c_{\mathrm{cipro}}, r_{\mathrm{SulA}}) - \cos(c_{\mathrm{bg}}, r_{\mathrm{SulA}}).
$$

**Decision use.** The panel keeps the stress references in view as weak external landmarks without letting them drive selection.

**Limits.** Reference alignment stays well below assay-era evidence. Poor alignment does not automatically invalidate a candidate X. Anchor-only reference comparisons remain especially fragile because the carried controls are length-mismatched to the dominant 60 bp synthetic cohort.

### representation_scree_diagnostic | PCA variance-decay diagnostic

#### Plot details

**Data.** Sample-based PCA reducer summaries for each candidate representation.

**Preprocessing.** Uses the persisted sampled reducer summaries directly.

**Definition.** Each panel shows retained variance ratios and cumulative retained variance over the stored principal components.

**Decision use.** The scree panels expose the representation-health gate directly.

**Limits.** Scree shape alone does not say whether the retained directions are useful for later supervised modeling.
