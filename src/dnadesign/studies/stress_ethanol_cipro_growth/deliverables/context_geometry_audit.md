# Context geometry audit

This deliverable asks whether candidate representations stay scientifically usable when the same promoter is read as `anchor_60bp` versus `full_context_1kb`. It is a secondary but important guardrail after the reference-margin analysis.

## Why this deliverable exists

A representation can look attractive on biology-facing evidence and still be unstable when extra sequence context is added. This deliverable makes that tradeoff visible before a candidate is treated as trustworthy.

## Plot guide

Read the paired context-shift plane first, then the metric distributions, then the compact summary metrics. Harmless scaffold shift usually means a consistent offset with limited subgroup breakup. Worrying representation change usually means larger subgroup-specific movement, heavier tails, weaker neighbor preservation, or movement that changes the biology-facing interpretation.

### context_shift_reference_plane | Context-shift reference plane

#### Why this plot exists

This plot keeps the anchor-only versus full-context comparison explicit at the level of paired promoter points in the reference-margin plane.

#### How to read it

Look for whether the context-expanded points move in a small, consistent way or whether they scatter into a different geometry. Small, coherent movement that preserves the basic relative ordering is more reassuring than large, subgroup-specific displacement.

#### What would worry us

Large rotations, subgroup-dependent drift, or movement that changes which promoters sit near the relevant reference surfaces are the main warning patterns. A context shift that pushes apparent winners toward the wrong biological interpretation is especially concerning.

#### Limits / guardrails

This plot shows paired movement, not proof of robustness. It also does not prove mechanism: pooled full-context vectors can reflect broader scaffold changes rather than anchor-local biology.

#### What to look at next

Use `context_delta_distributions` to see whether the center, spread, and tails support the visual impression from the paired plane.

### context_delta_distributions | Context-delta distributions

#### Why this plot exists

This plot turns the paired context shifts into explicit metric distributions so the reader can inspect center, spread, tails, and subgroup separation directly.

#### How to read it

Read the center of each distribution as the typical context effect, the spread as heterogeneity across promoters, the tails as the risk of large context failures, and subgroup separation as evidence that some promoter families respond differently from others. The important question is whether the distributions stay tight enough that the biology-facing comparisons remain interpretable.

#### What would worry us

Heavy tails, strongly shifted centers, or clear subgroup bifurcation are warning patterns. A small average shift with a long harmful tail is still a problem if it can move the promoters that matter most.

#### Limits / guardrails

These panels summarize distributions; they do not show per-pair trajectories. Read them together with the paired plane rather than treating them as a replacement.

#### What to look at next

Use `context_geometry_summary` for the compact candidate-by-candidate comparison once the raw distributions look reasonable.

### context_geometry_summary | Context-geometry summary

#### Why this plot exists

This plot pulls the most useful context-stability metrics into one compact candidate comparison so the reader can see whether the same representation still looks acceptable once stability is considered.

#### How to read it

Read the metrics as a set: self-cosine describes within-promoter alignment across contexts, shift magnitude describes the absolute size of the change, and neighborhood or geometry-preservation metrics describe whether local relationships survive the context change. Together they answer whether the candidate is merely moving or actually losing structure.

#### What would worry us

Candidates that look strong on the biology-facing deliverable but weak here deserve skepticism. Low self-cosine, large shift magnitude, and weak neighborhood preservation together are the clearest warning combination.

#### Limits / guardrails

This is a summary surface, not a hidden score. It compresses several stability questions into one view, so the labels and preferred directions matter more than any single bar ranking.

#### What to look at next

Move to `representation_tradeoff_scatter` to compare biology-facing evidence against these stability costs directly.
