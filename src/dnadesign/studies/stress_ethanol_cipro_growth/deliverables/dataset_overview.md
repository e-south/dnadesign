# Dataset Overview

The comparison set has `157,164` promoters: `157,160` DenseGen synthetic promoters and `4` carried controls. The figure defines the shared denominator and the annotation axes used in the later geometry summaries.

The current shared population is `N = 157,164`: `157,160` DenseGen synthetic promoters plus `4` carried controls. The view name `anchor_60bp` covers the dominant synthetic cohort. The carried controls are 35 bp, 165 bp, 200 bp, and 220 bp anchors.

### dataset_overview | Dataset inventory by cohort dimension

#### Plot details

**Data.** The anchor and 1 kb construct-context datasets contain the same promoter population. The inventory is therefore reported once across provenance, design family, regulator composition, Sigma-35 variant, and realized spacer length.

**Preprocessing.** None beyond the persisted cohort inventory table.

**Definition.** Each subpanel reports

$$
\mathrm{fraction}(c) = \frac{n_c}{157{,}164}.
$$

**Decision use.** The inventory defines the denominator and the annotation axes carried into the later summaries: design family, regulator composition, Sigma-35 variant, and spacer length.

**Limits.** Inventory balance is descriptive only. It does not measure representation quality or downstream usefulness.
