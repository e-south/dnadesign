# Representation comparison

This deliverable is the compact comparison surface for candidate representations. It is where the biology-facing evidence and the context-stability costs are read together without collapsing them into a hidden winner score.

## Why this deliverable exists

The study needs a way to compare candidate spaces directly after the reader has already seen the biology and context audit surfaces. This deliverable keeps those tradeoffs explicit and traceable.

## Plot guide

Read the tradeoff scatter after the reference-margin and context-audit deliverables. The axes were chosen to put biology-facing evidence, neighborhood evidence, and context stability into named comparisons rather than one opaque aggregate score.

### representation_tradeoff_scatter | Representation tradeoff scatter

#### Why this plot exists

This plot puts the candidate representations on explicit tradeoff axes so the reader can see which spaces preserve biology-facing evidence without paying an unacceptable stability cost.

#### How to read it

Read each panel by its named axes. The point of the figure is not to compute a total score; it is to show where a candidate sits when one evidence family is weighed against another. Use the visible labels to keep each point tied to its model family, sequence scope, and representation family.

#### What would worry us

Candidates that look good on one axis only, or that require a large stability penalty to gain modest biology-facing evidence, are not convincing. A visually isolated point is not enough if the underlying axes say it wins by exploiting only one summary.

#### Limits / guardrails

This is a comparison aid, not a final authority. It is intentionally not a hidden winner engine, and it should not replace the underlying deliverables that generated the plotted summaries.

#### What to look at next

Use `representation_scree_diagnostic` to check whether an apparently strong candidate is also geometrically healthy rather than pathologically low-rank.
