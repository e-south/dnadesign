# Dataset overview

This deliverable is the trust gate for everything that follows. It answers whether the checked-in anchor-only and full-context populations are symmetric enough that later comparison plots are worth reading.

## Why this deliverable exists

Later plots are only persuasive if they are built on comparable populations. This deliverable keeps the population accounting visible before any geometry or candidate-comparison surface is interpreted.

## Plot guide

Read the dataset overview first. If scope coverage, controls, or reference cohorts are visibly unbalanced here, keep that caveat in mind when reading every downstream plot.

### dataset_overview | Dataset overview

#### Why this plot exists

This plot makes the study inventory concrete. It shows whether the declared cohorts, controls, and reference promoters are all present across the anchor-only and full-context branches.

#### How to read it

Look for the same major families, regulator compositions, sigma-axis groups, and source classes on both scopes. Confirm that the expected controls and wildtype references are present before treating downstream comparison patterns as trustworthy.

#### What would worry us

Missing or sharply asymmetric cohorts are the main failure mode. If one scope is underrepresented, or if `spyP`, `sulAp`, or `J23105` coverage looks incomplete, later plots may still be descriptive but they stop being a clean apples-to-apples comparison.

#### Limits / guardrails

This plot is inventory only. It does not prove model readiness, biological separation, or representation quality.

#### What to look at next

Move to `reference_margin_gallery_wildtype` in the reference-margin analysis deliverable.
