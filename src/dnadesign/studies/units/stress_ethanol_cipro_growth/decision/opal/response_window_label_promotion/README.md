---
id: stress-ethanol-cipro-growth-response-window-label-promotion
title: Response-window label promotion
owner: stress_ethanol_cipro_growth
status: active
last_verified: 2026-07-15
first_hop: publisher.py
---

# Response-Window Label Promotion

This adapter has one job: translate a verified, approved study observation
bundle into OPAL's immutable observed-label contract.

It does not read Reader outputs, resolve aliases, combine experiments, calculate
RMF, fit a model, choose a selection view, or authorize synthesis. Those concerns
remain with Reader, the study observation package, OPAL objectives and models,
and study governance respectively.

The publisher verifies candidate IDs and sequence digests against the target
OPAL candidate table and binds the promotion to that complete `records.parquet`
snapshot, including the configured X column. It then atomically publishes a
versioned directory:

```text
_opal/response_window_labels_v1/
  observed_labels.parquet
  study_provenance.json
  promotion.manifest.json
```

`observed_labels.parquet` contains exactly `id`, `observed_round`, `batch_id`,
`y_space`, and one-dimensional `y_obs[8]`. The promotion manifest uses
`opal.observed_label_promotion.v1`. OPAL verifies the candidate/X snapshot,
study provenance, and labels before every read, and generic `ingest-y` cannot
mutate it.

The study provenance also records every measured candidate that is absent from
the promoted label table and the study-issued reason for that disposition. The
campaign must project that exact ID-and-reason set through the
`candidate_id_exclusion` eligibility rule. Publication and later verification
reject missing, extra, stale, or differently reasoned entries. The observation
bundle remains the authority; the campaign configuration is only its selection
projection.

The campaign uses `error_on_duplicate`; duplicate candidate/round labels are a
contract violation rather than a last-write-wins event.

The verified v1 publication contains 27 exact candidate labels. It records
eight measured-candidate exclusions, including the four repeated candidates
with unresolved source disagreement. Eight other repeated candidates use their
reviewed selected source. This publication makes no equal-experiment or
population-level uncertainty claim.

The explicit operator surface for a new approved version is:

```bash
uv run python -m \
  dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_window_label_promotion \
  publish \
  --observation-bundle <approved-observation-bundle> \
  --dataset-root src/dnadesign/usr/datasets/usr_prom_eth_cip_opal_candidates

uv run python -m \
  dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_window_label_promotion \
  verify \
  --dataset-root src/dnadesign/usr/datasets/usr_prom_eth_cip_opal_candidates
```

The publisher has no overwrite flag. The verify command is read-only.

Publication is create-only. An existing promotion directory is immutable. A
new approved corpus must use a new versioned directory and an explicit campaign
binding update; the publisher has no overwrite mode.

No production artifact can be published while the study observation policy or
any repeat decision remains under review. The study observation publisher also
fails closed when an included primary component is a lower, upper, or
indeterminate bound. Label promotion cannot reinterpret that finite bound as an
exact `y_obs` value.
