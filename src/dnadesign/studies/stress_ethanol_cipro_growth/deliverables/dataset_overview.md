# Dataset Overview

The shared comparison set contains `157,164` promoters: `157,160` DenseGen synthetic promoters and `4` carried controls. This figure defines the denominator used by the downstream summaries and reports the cohort inventory shown here: provenance, generation plan, and Sigma-35 variant.

The view name `anchor_60bp` still covers the dominant synthetic cohort. The carried controls are 35 bp, 165 bp, 200 bp, and 220 bp anchors.

### dataset_overview | Dataset inventory by cohort dimension

#### Plot details

**Data.** The anchor and 1 kb construct-context datasets contain the same promoter population. The inventory is therefore reported once across provenance, generation plan, and Sigma-35 variant.

**Preprocessing.** None beyond the persisted cohort inventory table.

**Definition.** Each subpanel reports

$$
\mathrm{fraction}(c) = \frac{n_c}{157{,}164}.
$$

**Decision use.** The inventory defines the denominator and the displayed cohort partitions carried into this review surface.

**Limits.** Inventory balance is descriptive only. It does not measure representation quality or downstream usefulness.
