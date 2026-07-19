---
id: opal-campaign-secg-msrb-greedy
title: SECG MSRB greedy campaign
owner: stress_ethanol_cipro_growth
status: active
last_verified: 2026-07-18
surface: opal_campaign
campaign_slug: secg_msrb_greedy
campaign_kind: study
runtime_status: round0_complete
---

## SECG MSRB Greedy Campaign

This campaign fits one shared eight-output model of the Reader response-window
phenotype and evaluates three Multistate Response Behavior (MSRB) views:
ethanol, ciprofloxacin, and AND. Each view ranks six candidates by
`behavior_score`. A round-robin allocator advances to the next unallocated
sequence when views overlap, so the physical batch contains 18 unique
sequences.

The model predicts
`[r00, r10, r01, r11, b00, b10, b01, b11]`; it does not predict MSRB directly.
The same predicted phenotype is interpreted under each study-issued target
mask. MSRB is the objective and `top_n` is the selector.

In the family landscape, farther right means better response ordering, farther
up means stronger intended-ON signal, and redder means stronger intended-OFF
suppression. All three directions matter. Measured controls provide context but
are not members of the unmeasured prediction pool, and sequence deduplication
can advance a later view to its next-best unallocated candidate. The
selected-candidate decomposition, not either scatter axis alone, explains a
rank.

The OFF-suppression color scale is a campaign-pinned symmetric linear display
at the prediction-pool absolute 99th percentile. Rectangular colorbar extensions
identify saturated tail colors without implying another score category; all
candidates and exact values remain available.

The campaign is a prospectively frozen greedy learning probe. Retrospective
ordering is weak, prospective hill-climb efficacy is unknown, and campaign
selection does not authorize synthesis. The
[study protocol](../../../studies/units/stress_ethanol_cipro_growth/decision/opal/multistate_response_behavior/README.md)
owns those scientific boundaries.

Round 0 completed on 2026-07-18 with 27 measured labels, 154,785 scored
candidates, six allocations per view, and 18 sequence-unique candidates after
one cross-view overlap was replaced by AND's next unallocated rank. Output
verification found zero score or membership mismatches. These checks establish
runtime and artifact fidelity, not prospective enrichment or synthesis
readiness.
