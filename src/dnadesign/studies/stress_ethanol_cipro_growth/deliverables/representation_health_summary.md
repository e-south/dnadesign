# Representation Health Summary

Candidate spaces are screened for collapse with effective rank, PC1 variance
fraction, and cosine-distance spread. Read the bars directly by metric
magnitude rather than a pass/fail hue. This is an eligibility gate: it asks
whether a stored view has enough non-degenerate geometry to interpret, not
whether that view is biologically best.

### representation_health_summary | Representation health summary

#### Plot details

**Data.** Each candidate summary combines a stratified candidate sample with the
persisted PCA reducer summary for that same sampled view. The current candidate
reducers fit roughly 2k sampled rows and retain 16 PCA components. The direct
comparison includes the six first-class intermediate-embedding views and the
six matching output-layer mean views. Reference rows remain controls inside
preserved samples rather than independent effective-rank cohorts.

The intermediate views are row-level summaries from Evo2 7B block26 hidden
states: anchor-source `seq_mean`, forward full-context `seq_mean`, forward
full-context `anchor_mean`, reverse-complement full-context `seq_mean`,
reverse-complement full-context `anchor_mean`, and the equal-block forward/RC
`anchor_mean` concat. The output-layer views follow the same pooling scopes but
use mean-pooled per-token logits as diagnostic surfaces.

**Preprocessing.** Pairwise cosine metrics use view-level standardization
followed by row L2 normalization. Study builders use the collapse-tolerant
path: zero-variance columns are set to `0.0` after scaling, and zero-norm rows
remain `0.0` so degenerate spaces stay finite and are exposed by the health
gate. PCA-derived metrics come from the stored reducer summaries on the sampled
rows. The health plot sees only stored row vectors; it does not inspect the
original Evo2 token grid directly.

**Definition.** The primary rank-health statistic is retained-component effective rank,

$$
r_{\mathrm{eff}} = \exp\left(-\sum_i p_i \log p_i\right),
$$

where

$$
p_i = \frac{\lambda_i}{\sum_j \lambda_j}.
$$

The panel also reports explained variance captured by the retained reducer components, the retained PC1 variance fraction,
\(\lambda_1 / \sum_j \lambda_j\), the median pairwise cosine distance, and the
pairwise cosine-distance interquartile range. All pairwise cosine distances are
computed after the shared view-level standardization and row-normalization
contract.

**Decision use.** Collapsed spaces drop out before design structure or Sigma-35
ordering are compared. Higher retained rank means variance is spread across
more retained PCA directions, not that the full original embedding has that
exact rank. Selection still depends on design structure, Sigma-35 ordering,
reference behavior, and context/orientation robustness.

**Limits.** Passing the health gate does not mean a space is
biologically aligned. A high-capacity space can still be dominated by nuisance
factors such as spacer length or construct context.
