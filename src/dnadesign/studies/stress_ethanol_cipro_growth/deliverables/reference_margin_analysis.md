# Reference-margin analysis

This deliverable is the primary biology-facing readout. It asks whether candidate representations place promoters closer to the expected wildtype references relative to `J23105`, and whether that signal survives outside the 2D margin plane.

## Why this deliverable exists

The main scientific question is not whether a panel looks visually separated. It is whether candidate spaces preserve the reference-relative biology we care about. This deliverable keeps that question explicit and keeps the notebook focused on sanctioned evidence rather than duplicate summary surfaces.

## Plot guide

Read the wildtype reference-margin gallery first, then the reference-neighbor evidence. Treat the synthetic-centroid gallery as appendix-only proxy context. `dual_margin_plane` is intentionally not a separate sanctioned surface because it repackages the same wildtype-margin evidence rather than adding an independent readout.

### reference_margin_gallery_wildtype | Wildtype reference-margin gallery

#### Why this plot exists

This is the core biology-facing surface. It shows each candidate representation in the ethanol-versus-ciprofloxacin margin plane, with both axes defined relative to `J23105`.

#### How to read it

Read movement toward the upper-right as movement toward both wildtype references relative to the control baseline. The key comparison is not raw cosine or cluster shape; it is the signed margin relative to `J23105`, because that keeps the interpretation anchored to the shared control promoter.

#### What would worry us

Representations that only improve one axis, collapse the full population into an uninformative cloud, or move controls and intended promoters in the same direction are not reassuring. Large apparent separation with weak reference-relative ordering is also a warning sign.

#### Limits / guardrails

This is a descriptive 2D projection of explicit reference-relative metrics. It does not prove class separability, mechanism, or downstream task success on its own.

#### What to look at next

Check `reference_neighbor_evidence` to see whether the high-dimensional neighborhoods agree with the 2D margin story.

### reference_neighbor_evidence | Reference-neighbor evidence

#### Why this plot exists

This plot asks whether the intended wildtype references actually appear in the local neighborhoods of candidate promoters in the full candidate space. It is the main sanity check that the margin-plane readout is not just a 2D artifact.

#### How to read it

Read higher reference-neighbor hit rates as better evidence, and read lower reference-neighbor ranks as better evidence. The metric labels matter: some panels are higher-is-better and some are lower-is-better, so use the axis text rather than bar height alone as the interpretation rule.

#### What would worry us

If a candidate looks strong in the margin plane but fails to place the relevant references in its local neighborhoods, the biology-facing story is weaker than it first appears. Flat or non-separating neighbor metrics across all candidates are also a warning that this run may not distinguish the spaces meaningfully.

#### Limits / guardrails

This is still summary evidence. It is stronger than a 2D-only panel because it comes from the full candidate space, but it is not a hidden winner score and it does not replace downstream judgment.

#### What to look at next

Move to `context_shift_reference_plane` to see whether the same candidates remain stable when sequence context expands from anchor-only to full-context.

### reference_margin_gallery_synthetic_centroids | Synthetic-centroid reference-margin gallery

#### Why this plot exists

This appendix plot keeps a proxy view of reference-relative behavior that can still be useful for orientation when the reader wants one more coarse comparison surface.

#### How to read it

Read it after the wildtype plots, not before them. Use it as a proxy view that may help explain broad tendencies, not as equal evidence to the wildtype-reference analysis.

#### What would worry us

If the synthetic-centroid proxy materially disagrees with the wildtype-reference surfaces, trust the wildtype-reference surfaces. A proxy that tells a cleaner story than the primary evidence is a sign to slow down, not a reason to upgrade the proxy.

#### Limits / guardrails

This is appendix-only. It is a companion audit surface, not a primary decision surface, and it does not replace wildtype-relative evidence.
