# Representation Health Summary

Candidate spaces are screened for collapse with effective rank, PC1 variance fraction, and cosine-distance spread. Bar color marks the gate status: `pass`, `warn`, or `fail`.

### representation_health_summary | Representation health summary

#### Plot details

**Data.** Each candidate summary combines a stratified candidate sample with the persisted PCA reducer summary for that same sampled view.

**Preprocessing.** Pairwise cosine metrics use view-level standardization followed by row L2 normalization. PCA-derived metrics come from the stored reducer summaries on the sampled rows.

**Definition.** The primary rank-health statistic is

$$
r_{\mathrm{eff}} = \exp\left(-\sum_i p_i \log p_i\right),
$$

where

$$
p_i = \frac{\lambda_i}{\sum_j \lambda_j}.
$$

The panel also reports the retained PC1 variance fraction,
\(\lambda_1 / \sum_j \lambda_j\), the median pairwise cosine distance, and the
pairwise cosine-distance interquartile range. All pairwise cosine distances are
computed after the shared view-level standardization and row-normalization
contract.

**Decision use.** Collapsed spaces drop out before design structure or Sigma-35 ordering are compared. Higher rank means variance is spread across more directions. Selection still depends on design structure and context robustness.

**Limits.** Passing the health gate does not mean a space is
biologically aligned. A high-capacity space can still be dominated by nuisance
factors such as spacer length or construct context.
