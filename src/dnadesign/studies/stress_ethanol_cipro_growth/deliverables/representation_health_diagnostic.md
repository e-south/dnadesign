# Representation health diagnostic

This diagnostic checks whether each candidate space still carries usable variance structure after centering and reduction. It is a rank-health panel that should be read alongside the evidence and context-stability plots.

### representation_scree_diagnostic | PCA variance-decay diagnostic

#### Plot details

**Data.** Each curve summarizes the PCA explained-variance decay for a candidate representation space after centering and reduction.

**Definition.** Let \(p_i\) be the normalized explained-variance ratio for principal component \(i\). The effective rank is

$$
\mathrm{effective\_rank}
=
\exp\left(
-\sum_i p_i \log p_i
\right).
$$

**Interpretation.** A steep early decay means a small number of principal components explain most of the variance. Low effective rank indicates concentrated variance support after centering and reduction. `effective_rank` is a rank-health diagnostic, not a decision rule by itself.
