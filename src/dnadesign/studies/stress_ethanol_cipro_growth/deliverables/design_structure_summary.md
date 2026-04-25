# Design Structure Summary

The summary measures how clearly each candidate space preserves design family, regulator composition, Sigma-35 variant, and spacer length.

### design_structure_summary | Design-structure summary

#### Plot details

**Data.** Each candidate is evaluated on a stratified sample over the shared promoter population with trusted metadata axes: `design_family`, `design_regulator_composition`, `sig35_variant`, and `spacer_length`.

**Preprocessing.** All cosine geometry uses view-level standardization followed by row L2 normalization. Study builders use the collapse-tolerant path: zero-variance columns are set to `0.0` after scaling, and zero-norm rows remain `0.0`.

**Definition.** For each annotation axis \(A\), the cohort centroid is

$$
c_g = \mathrm{normalize}\left(\frac{1}{|g|}\sum_{i \in g} z_i\right),
$$

and the reported separation statistic is

$$
S_A = \frac{B_A}{W_A + \epsilon},
$$

with

$$
B_A = \mathrm{mean}_{g<h}\left(1 - \cos(c_g, c_h)\right)
$$

and

$$
W_A = \mathrm{mean}_g \ \mathrm{mean}_{i \in g}\left(1 - \cos(z_i, c_g)\right).
$$

The design-family panel is reported both unadjusted and after balancing by
Sigma-35 variant and spacer length.

**Decision use.** The reported ratios compare how well each candidate space preserves the design structure already present in the study.

**Limits.** The ratio measures annotation separation across known study labels. Strong spacer-length separation can reflect nuisance structure, and coarse cohort separation can still miss useful within-cohort ordering.

### balanced_design_family_margin_gallery | Balanced design-family margin gallery

#### Plot details

**Data.** The gallery uses the main 7B intermediate candidates and shows the full shared promoter population. Only the centroid directions are balanced so the explanatory axes are not driven by Sigma-35 or spacer imbalance.

**Preprocessing.** All margins are computed after view-level standardization and row L2 normalization. The plotted population stays full, while the underlying design-family centroid directions are computed from balanced synthetic subsets over `design_family`, `sig35_variant`, and `spacer_length`, with leave-one-out handling for in-cohort rows that participate in those balanced centroids.

**Definition.** The x axis is the ethanol-versus-background margin

$$
m_{\mathrm{eth}}(x) = \cos(z_x, c_{\mathrm{eth}}) - \cos(z_x, c_{\mathrm{bg}})
$$

and the y axis is the ciprofloxacin-versus-background margin

$$
m_{\mathrm{cipro}}(x) = \cos(z_x, c_{\mathrm{cipro}}) - \cos(z_x, c_{\mathrm{bg}}).
$$

Positive values mean the row is closer to the target family centroid than to background on that axis.

**Decision use.** This is the row-level bridge for the balanced design-family separation ratio. It shows whether the family quadrants remain visible after the centroid directions are balanced against the nuisance axes used by the scalar summary.

**Limits.** The gallery is still annotation-conditioned and two-dimensional. It helps explain the balanced separation metric, but it does not replace that scalar or imply phenotype.
