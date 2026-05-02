# Representation Tradeoff

This synthesis plot summarizes the full pre-assay ladder without replacing it. It is the publication-facing “why this X” figure for the current synthetic-only comparison surface.

### candidate_decision_frontier | Representation tradeoff

#### Plot details

**Data.** Each point is one healthy study-facing candidate representation from the trimmed 7B inventory. The plot joins persisted summary tables for representation health, balanced design-family separation, Sigma-35 ordinal structure, and context robustness.

**Preprocessing.** This figure does not recompute row-level geometry. It reuses the persisted scalar summaries, each of which already follows the shared contract of view-level standardization plus row L2 normalization before cosine-based metrics are formed.

**Definition.** Point size is retained-component effective rank from the sampled PCA reducer summary,

$$
r_{\mathrm{eff}} = \exp\!\left(-\sum_i p_i \log p_i\right),
\qquad
p_i = \frac{\lambda_i}{\sum_j \lambda_j},
$$

where the \(\lambda_i\) values are the positive explained-variance ratios among retained PCA components. In plain language, a larger point means retained variance is spread across more sampled PCA directions rather than collapsing into one dominant retained component.

The x axis is the balanced design-family separation ratio,

$$
S_{\mathrm{design}}^{\mathrm{balanced}}
=
\frac{\operatorname{mean}\!\left(d_{\mathrm{between}}\right)}
     {\operatorname{mean}\!\left(d_{\mathrm{within}}\right)},
$$

and the y axis is the global Sigma-35 ordinal Spearman,

$$
\rho_{\mathrm{sig35}}
=
\operatorname{Spearman}\!\left(\Delta_{\mathrm{expected}}, \Delta_{\mathrm{observed}}\right),
$$

with one directly labeled point per candidate. In plain language, farther right means the candidate keeps the balanced design families more separated than their internal spread, and higher means the candidate preserves the intended Sigma-35 ladder more faithfully.

**Decision use.** Read this as a synthesis of the healthy-candidate ladder rather than a replacement for it. The plot is strongest when it confirms what the gate, Sigma-35 audit, and anchor-vs-context shift summaries already say independently. A candidate is more convincing when it sits farther right and higher up while also keeping a reasonably large retained-rank point, because that combination suggests preserved structure without obvious collapse in the retained PCA diagnostic.

**Limits.** A two-axis synthesis still compresses a multi-panel decision process. The context story is no longer encoded directly here, so this plot should be read after the row-level bridge plots and the direct context comparison rather than in isolation.
