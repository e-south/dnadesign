# Dataset Overview

The shared comparison set contains `157,279` promoter rows. The current
provenance split is `157,183` DenseGen-derived rows, `38` manual or wild-type
reference/control rows, and `58` synthetic reference-standard rows. This figure
defines the denominator used by the downstream summaries and reports the cohort
inventory shown here: provenance, generation plan, and Sigma-35 variant.

The merged anchor-source insert view intentionally carries mixed lengths. The
dominant DenseGen cohort is 60 bp, while native references, controls, and some
standards preserve their source insert or derived analysis-window lengths. This
is why "anchor_60bp" is a convenient study label, not a literal invariant for
every row in the merged source.

### dataset_overview | Dataset inventory by cohort dimension

#### Plot details

**Data.** The anchor and 1 kb construct-context datasets contain the same promoter population. The inventory is therefore reported once across provenance, generation plan, and Sigma-35 variant.

**Preprocessing.** None beyond the persisted cohort inventory table.

**Definition.** Each subpanel reports

$$
\mathrm{fraction}(c) = \frac{n_c}{157{,}279}.
$$

**Decision use.** The inventory defines the denominator and the displayed cohort partitions carried into this review surface.

**Limits.** Inventory balance is descriptive only. It does not measure representation quality or downstream usefulness.
