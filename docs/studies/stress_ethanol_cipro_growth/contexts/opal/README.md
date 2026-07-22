---
id: stress-ethanol-cipro-growth-opal-context-index
title: Stress OPAL context
owner: dnadesign-maintainers
status: active
last_verified: 2026-07-22
audience:
  - operator
  - agent
---

## Stress OPAL Context

`secg_msrb_greedy` is the sole executable stress-study campaign and uses
`multistate_response_behavior_v1`. SFXI and RMF remain source or comparator
evidence. Start with the [study status](../../record/status.md), the [MSRB assay
binding](multistate-response-behavior.md), or [read-only campaign
verification](../../routes/decision/opal/campaign-commands.md#read-only-campaign-verification).

### Current campaign

- [Candidate table](candidate-table.md): shared USR candidate-table and label
  source contract.
- [MSRB study application](multistate-response-behavior.md): assay binding and
  campaign interpretation. Focused routes cover the [Reader-to-label
  path](multistate-response-behavior.md#end-to-end-evidence-path),
  [soft-min scale](multistate-response-behavior.md#what-the-soft-min-scale-changes),
  [uncertainty](multistate-response-behavior.md#uncertainty-and-censoring), and
  [claim limits](multistate-response-behavior.md#claim-boundaries).
- [MSRB symbol walkthrough](multistate-response-behavior-walkthrough.html):
  browser-native explanation of the symbols, behavior families, soft minimum,
  and compensation scale.

### Comparator evidence

- [SFXI round-0 source evidence](sfxi-round0-source-evidence.md): declared
  SFXI vec8 inputs, source runs, and immutable artifact provenance.
- [Response assay and objective comparison](response-metastudy.md): read-only review of the
  SFXI source ledgers, Reader response-window summaries, RMF/MSRB comparison,
  label truth, and predictor support.
- [Response-Magnitude Feasibility (RMF)](response-magnitude-feasibility.md):
  thresholded requirements diagnostic and frozen comparator evidence. RMF is
  not an executable stress campaign; its round-0 record remains in the
  metastudy.

### Separate in-silico probes

- [DenseGen TFBS learnability probe v1](densegen-tfbs-learnability-probe-v1.md):
  study-owned v1 contract for scalar TF family content and slot-position
  synthetic-control campaigns. Realized profile boundaries live in the
  source package README and profile registry.
- [DenseGen motif QA K12/S3 v1](densegen-motif-qa-k12-s3-v1.md): K12,
  three-seed, trajectory-based motif-composition QA benchmark and
  execution reference.
- [DenseGen axis probe v0](densegen-axis-probe-v0.md): scratch-only
  K6 synthetic-oracle probe for OPAL/LatentDNA readiness.
