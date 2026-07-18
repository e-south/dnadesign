---
id: secg-msrb-study-protocol
title: Stress-promoter MSRB protocol
owner: stress_ethanol_cipro_growth
status: active
last_verified: 2026-07-18
---

## Stress-Promoter MSRB Protocol

`protocol.yaml` is the machine-readable study decision for the active
Multistate Response Behavior (MSRB) learning probe. It fixes the response-window
Y contract, state order, target masks, assay-resolution normalization, label
policy, model target, selector, allocation rule, evidence posture, and claim
boundaries.

The didactic scientific contract is
[`docs/studies/stress_ethanol_cipro_growth/contexts/opal/multistate-response-behavior.md`](../../../../../../../../../../docs/studies/stress_ethanol_cipro_growth/contexts/opal/multistate-response-behavior.md).
OPAL owns the generic objective equations; this directory owns only the
stress-study choices applied to those equations.

The campaign is a prospectively frozen greedy learning probe. Its purpose is to
measure whether sequence-to-phenotype predictions and MSRB enrichment improve
as observations accumulate. It does not authorize synthesis by itself and does
not claim that retrospective prediction support is strong.

`activation_audit.json` records the bounded activation decision and binds the
exact protocol, objective, tests, documentation, campaign, and review-plot
configuration by SHA-256. The receipt includes the protocol digest, while the
protocol names the receipt schema and path without copying the receipt digest.
That one-way relationship prevents a self-referential digest cycle.

The receipt does not overwrite the shadow evaluation. The shadow decision
remains the evidence that retrospective results did not justify claiming
superior hill-climbing or synthesis readiness. The active decision authorizes
only a prospectively frozen learning probe under the separately declared study
protocol.
