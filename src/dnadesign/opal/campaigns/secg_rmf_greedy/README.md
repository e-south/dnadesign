---
id: opal-campaign-secg-rmf-greedy
title: SECG RMF greedy campaign
owner: stress_ethanol_cipro_growth
status: active
last_verified: 2026-07-15
surface: opal_campaign
campaign_slug: secg_rmf_greedy
campaign_kind: study
runtime_status: round0_complete
---

## SECG RMF Greedy Campaign

**Owner:** stress_ethanol_cipro_growth study
**Lifecycle:** round 0 complete; prospective assay pending
**Last verified:** 2026-07-15

This study campaign fits one shared eight-output response model and evaluates
three RMF selection views: ethanol, ciprofloxacin, and AND. Each view nominates
six candidates. The declared round-robin allocator advances a view to its
next-best unallocated sequence when preferred sets overlap, producing an exact
18-sequence batch while retaining the overlap and replacement trace.

Round 0 completed as run `r0-2026-07-16T01:32:16+00:00` from 27 exact,
manifest-pinned Reader response-window labels. Each view received six slots.
One candidate appeared in two preferred lists, so the AND view advanced once
to the next unallocated sequence and the final batch contains 18 unique
sequences.

This run is a prospectively frozen learning probe. `model_support_ready`
remains false, and selection does not authorize synthesis. The
[stress-study OPAL route](../../../../../docs/studies/stress_ethanol_cipro_growth/routes/decision/opal/README.md)
owns scientific readiness and handoff status.

SFXI round-0 runs remain study evidence in their declared y-space. They are not
executable routes into this campaign.
