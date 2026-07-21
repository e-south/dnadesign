---
id: secg-msrb-study-protocol
title: Stress-promoter MSRB protocol
owner: stress_ethanol_cipro_growth
status: active
last_verified: 2026-07-19
---

## Stress-Promoter MSRB Protocol

`protocol.yaml` is the machine-readable study decision for the active
Multistate Response Behavior (MSRB) learning probe. It fixes the response-window
Y contract, state order, target masks, shared soft-min scale, label policy,
model target, selector, allocation rule, evidence posture, and claim boundaries.

The generic objective source of truth is
[`src/dnadesign/opal/docs/plugins/objectives/multistate-response-behavior.md`](../../../../../../opal/docs/plugins/objectives/multistate-response-behavior.md).
The applied study contract is
[`docs/studies/stress_ethanol_cipro_growth/contexts/opal/multistate-response-behavior.md`](../../../../../../../../../../docs/studies/stress_ethanol_cipro_growth/contexts/opal/multistate-response-behavior.md).
OPAL owns the generic objective equations; this directory owns only the
stress-study choices applied to those equations.

The campaign is a prospectively frozen greedy learning probe. Its purpose is to
measure whether sequence-to-phenotype predictions and MSRB enrichment improve
as observations accumulate. It does not authorize synthesis by itself and does
not claim that retrospective prediction support is strong.

`evaluation_baseline.yaml` fixes how the first measured batch will be judged
before its outcomes are known. It binds the campaign configuration, allocator
version, prediction ledger, selected 18, aliases, and prior 27-label corpus.
The verifier recomputes MSRB from every predicted eight-value phenotype and
requires the sequence-unique allocation to reproduce the selected 18 exactly.
It also fixes the treatment of ties, even-sample medians, undefined rank
correlations, and missing values so the later comparison has one replayable
interpretation.

For each target view, all 296,010 possible groups of six prior labels form a
deterministic historical reference for the best and median observed MSRB. This
comparison needs no random seed and no extra plate. It is not a randomized
control from the current candidate universe and cannot by itself establish
acquisition efficacy or hill climbing. Verification does not change campaign
or synthesis state.

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

The shadow manifest and decision remain one generated workbench bundle. The
bundle verifier owns their complete inventory, byte, schema, provenance, and
derivation checks. The source-tree receipt records their paths and digests but
does not duplicate a partial bundle. The adversarial audit is packaged source
evidence and is checked byte-for-byte in a clean checkout. The active protocol
and activation receipt are also declared package data.
