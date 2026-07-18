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
