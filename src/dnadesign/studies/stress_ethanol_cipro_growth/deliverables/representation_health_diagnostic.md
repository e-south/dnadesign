# Representation health diagnostic

This deliverable checks whether candidate representations are geometrically healthy enough that the comparison plots are worth trusting. It is a diagnostic surface, not a separate decision engine.

## Why this deliverable exists

Biology-facing metrics are easier to trust when the underlying representation is not obviously collapsed. This deliverable keeps low-rank failure modes visible before they quietly distort the comparison.

## Plot guide

Read the scree diagnostic after the tradeoff scatter. A candidate that looks attractive on the tradeoff surface but collapses nearly all variance into a tiny number of components should be treated cautiously.

### representation_scree_diagnostic | Representation scree diagnostic

#### Why this plot exists

This plot summarizes explained-variance decay so the reader can see whether a representation still has usable dimensional structure after pooling and reduction.

#### How to read it

Look for whether the early components dominate too aggressively. A gradual scree curve with a reasonable effective-rank story is healthier than a curve that dumps most variance into the first few components immediately.

#### What would worry us

Early scree collapse is the clearest warning pattern. If the first few components explain almost everything, the representation may be acting like a much lower-rank object than the comparison plots suggest.

#### Limits / guardrails

This diagnostic does not decide the biology question on its own. It is a health check that should be read alongside the reference-margin and tradeoff surfaces.
