# Context Robustness Summary

The summary measures whether the 1 kb construct context preserves or dilutes the anchor-level design geometry.

### context_robustness_summary | Context robustness summary

#### Plot details

**Data.** The comparison uses a 4,096-row design-family-stratified sample of aligned anchor and construct-context rows, with one context per promoter anchor, grouped into the canonical representation families.

**Preprocessing.** Anchor and context vectors are standardized within view and row-normalized before any cosine or centroid-distance metric is computed on the persisted sample. Study builders use the collapse-tolerant path: zero-variance columns are set to `0.0` after scaling, and zero-norm rows remain `0.0`.

**Definition.** The row-level stability statistic is

$$
\mathrm{self\_cos}_i = \cos(z_i^{\mathrm{anchor}}, z_i^{\mathrm{context}}).
$$

For each annotation axis \(A\), the retention statistic is

$$
\mathrm{retention}_A =
\rho_{\mathrm{Pearson}}\left(\mathrm{upper}(D_A^{\mathrm{anchor}}),
\mathrm{upper}(D_A^{\mathrm{context}})\right),
$$

where each \(D_A\) is the cohort centroid-distance matrix in that view. The
panel reports retention for design family, regulator composition, and Sigma-35
variant.

**Decision use.** The reported metrics compare how well full-context pooling preserves the anchor-level cohort geometry.

**Limits.** This panel reports a sample estimate rather than the full aligned population. High anchor-context agreement can still preserve nuisance structure, and low agreement does not automatically make the anchor-only space the right choice.

### context_pair_summary | Direct context comparison

#### Plot details

**Data.** This summary reuses two densegen-only aligned 512-row design-family-stratified samples from the 7B intermediate family: anchor vs context anchor mean, and anchor vs full 1 kb seq mean.

**Preprocessing.** Anchor and context vectors are standardized by view and row-normalized before row-level self cosine and shift magnitudes are computed. The plot itself only aggregates those persisted row-level values to medians.

**Definition.** For each aligned pair family, the plotted metrics are the medians of

$$
\cos\left(z^{\mathrm{anchor}}, z^{\mathrm{context}}\right)
$$

and

$$
\left\|z^{\mathrm{anchor}} - z^{\mathrm{context}}\right\|_2.
$$

**Decision use.** This is the direct context proof: the chosen anchor-mean path should show higher median self-cosine and lower median L2 shift than whole-sequence 1 kb pooling on the same aligned rows.

**Limits.** Median summaries intentionally suppress subgroup shape and tails. Use the main context-robustness summary or the persisted row-level scalars when subgroup diagnostics are needed.
