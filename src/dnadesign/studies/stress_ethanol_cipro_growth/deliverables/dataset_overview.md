# Dataset inventory by cohort dimension

LatentDNA is a downstream comparison surface over the current `infer_batch_preparation` record. DenseGen remains the upstream source of cohort semantics. This deliverable fixes the denominator before any geometry or reference-margin plot is interpreted.

The current population is one shared promoter set: `157,160` DenseGen rows plus `4` manual or wildtype controls, for `N = 157,164`. The 60 bp anchor handoff and the 1 kb construct-context handoff contain the same promoters.

### dataset_overview | Dataset inventory by cohort dimension

#### Plot details

**Data.** This is one promoter population viewed through several cohort partitions. The denominator for each partition is \(N = 157{,}164\): \(157{,}160\) DenseGen designs plus \(4\) manual or wildtype controls. The 60 bp anchor handoff and the 1 kb construct-context handoff contain the same promoter population, so this plot does not facet by anchor versus context.

**Definition.** Each subpanel is a separate partition of the same \(N\) records. Within a subpanel, category fractions are computed as

$$
\mathrm{fraction}(c) = \frac{n_c}{157{,}164}.
$$

The categories in each subpanel should sum to one, up to rounding.

**Interpretation.** Read each subpanel independently. The plot answers whether the study population is balanced enough for downstream comparisons across provenance, generation plan, and Sigma-35 variant. It is not a stacked list of unrelated scalar counts.
