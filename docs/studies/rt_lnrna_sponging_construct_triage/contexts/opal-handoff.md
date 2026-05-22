---
doc_id: study-rt-lnrna-sponging-construct-triage-opal-handoff
surface: study-context
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-05-22
---

## OPAL Handoff

OPAL readiness means:

- one candidate universe with stable `id` values in a USR-shaped
  `records.parquet`;
- canonical base columns: `id`, `bio_type`, `sequence`, `alphabet`, and
  `length`;
- one explicit Arrow fixed-size-list vector `X` column selected by the study;
- a future label slot for `SpongingAssayObservation`;
- no learned `Y ~ X` campaign before real labels exist.

Pre-assay records may be OPAL-ready in shape, but OPAL `run` and `explain` are
blocked until a configured label source satisfies OPAL's own label contract.
For shared labels, prefer `labels.source.kind: usr_sidecar` and
`writeback.prediction_records: ledger_only`.

Abundance priors are allowed as metadata or separate analysis targets. They
must not be exported as `normalized_TF_sponging_label`.
