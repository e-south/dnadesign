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
