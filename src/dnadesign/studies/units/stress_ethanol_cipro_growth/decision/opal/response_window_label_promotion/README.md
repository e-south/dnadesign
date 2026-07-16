---
id: stress-ethanol-cipro-growth-response-window-label-promotion
title: Response-window label promotion
owner: stress_ethanol_cipro_growth
status: active
last_verified: 2026-07-16
first_hop: publisher.py
---

# Response-Window Label Promotion

This adapter has one job: translate a verified, approved study observation
bundle into OPAL's immutable observed-label contract.

It does not read Reader outputs, resolve aliases, combine experiments, calculate
RMF, fit a model, choose a selection view, or authorize synthesis. Those concerns
remain with Reader, the study observation package, OPAL objectives and models,
and study governance respectively.

The publisher verifies candidate IDs and sequence digests against the study's
candidate table and binds the promotion to the complete `records.parquet`
snapshot. Campaign binding is a separate check that verifies the configured X
column. The publisher then atomically creates a versioned directory:

```text
_opal/response_window_labels_v4/
  observed_labels.parquet
  source_observation.manifest.json
  study_provenance.json
  promotion.manifest.json
_opal/response_window_label_promotion.head.json
```

`observed_labels.parquet` contains `id`, the study-issued `display_label`,
`observed_round`, `batch_id`, `y_space`, and one-dimensional `y_obs[8]`. OPAL
preserves the display label as optional presentation metadata; it never uses the
label to resolve candidate identity or train the model. The promotion manifest uses
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

The event key is `(candidate ID, observed round)`. A duplicate event is a
contract violation. The same candidate may be measured again in a strictly
later round. `display_label` is presentation metadata and may change without
changing candidate identity. OPAL retains every event for review and applies
the campaign's declared `latest_only` policy when forming the training table.

A dataset-scoped lineage head serializes publication. Lineage genesis is
exactly a round-0 publication with no parent. Every later publication must name
the manifest recorded by the authoritative head; missing parents and stale
forks fail while the dataset lock is held. The publisher verifies that parent,
carries its rows forward, and appends one new study-issued batch. The new batch
must use a previously unused batch ID and a round later than every prior round.
Candidate/round duplication, round regression, batch reuse, candidate or
sequence drift, and non-finite or non-eight-component vectors fail before
publication.

The bundle copies the verified source observation manifest. Deep verification
checks its digest and exact policy/source claims, then checks label-event counts,
unique-candidate counts, rounds, batches, prior inventory, and candidate-table
claims against the verified artifacts. The operator surface reports
`label_event_count` and `unique_candidate_count` explicitly.

Candidate exclusions accumulate conservatively. A prior exclusion remains in
force unless the incoming approved batch supplies an exact label for that
candidate. A changed exclusion reason fails rather than rewriting prior study
provenance, and an incoming exclusion cannot contradict any promoted label.

The approved pre-round-0 source contains 27 exact candidate labels. It records
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
  --dataset-root src/dnadesign/usr/datasets/usr_prom_eth_cip_opal_candidates \
  --output-relative-directory _opal/response_window_labels_v4

uv run python -m \
  dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_window_label_promotion \
  verify \
  --dataset-root src/dnadesign/usr/datasets/usr_prom_eth_cip_opal_candidates \
  --output-relative-directory _opal/response_window_labels_v4

uv run python -m \
  dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_window_label_promotion \
  verify-campaign-binding \
  --dataset-root src/dnadesign/usr/datasets/usr_prom_eth_cip_opal_candidates \
  --output-relative-directory _opal/response_window_labels_v4 \
  --campaign-config src/dnadesign/opal/campaigns/secg_rmf_greedy/configs/campaign.yaml
```

The publisher has no overwrite flag. `verify` checks the study bundle without
coupling it to an OPAL campaign. `verify-campaign-binding` separately requires
an explicit campaign config and re-verifies its candidate ID and X-column
snapshot plus the exact exclusion projection. Both verification commands are
read-only. Run campaign binding verification only after the campaign config
explicitly names the new versioned label and promotion-manifest paths.

Publication is create-only. An existing promotion directory is immutable. A
new approved batch must use a new versioned directory, name the immediately
prior promotion, and update the campaign binding explicitly; the publisher has
no overwrite mode. Omit `--prior-promotion-manifest` only for the first
publication in a lineage.

No production artifact can be published while the study observation policy or
any repeat decision remains under review. The study observation publisher also
fails closed when an included primary component is a lower, upper, or
indeterminate bound. Label promotion cannot reinterpret that finite bound as an
exact `y_obs` value.
