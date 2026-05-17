# Design Structure Summary

The summary measures how clearly each stored candidate view preserves design
family, regulator composition, Sigma-35 variant, and spacer length. It compares
row-level representation summaries, not raw promoter strings: some rows come
from causal mean pooling over a source insert, some from causal mean pooling
inside a 1 kb context, and one view concatenates normalized forward and
reverse-complement context-anchor summaries.

### design_structure_summary | Design-structure summary

#### Plot details

**Data.** Each candidate is evaluated on a stratified sample over the shared
promoter population with trusted metadata axes: `design_family`,
`design_regulator_composition`, `sig35_variant`, and `spacer_length`. The
candidate labels identify how the row vector was formed: `seq_mean` averages
all emitted positions, `anchor_mean` averages the Construct-provided anchor
span, and the bidirectional-summary candidate concatenates equal-weight forward
and reverse-complement `anchor_mean` blocks.

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

**Decision use.** The reported ratios compare how well each candidate space
preserves the design structure already present in the study. The intended
winner is not the view with the largest rank; it is the view that remains
healthy while preserving the relevant synthetic-design axes.

**Limits.** The ratio measures annotation separation across known study labels.
Strong spacer-length separation can reflect nuisance structure, and coarse
cohort separation can still miss useful within-cohort ordering. This plot does
not estimate promoter activity.

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
