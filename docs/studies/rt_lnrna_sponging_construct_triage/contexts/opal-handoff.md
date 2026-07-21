---
doc_id: study-rt-lnrna-sponging-construct-triage-opal-training-dataset
surface: study-context
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-07-21
---

## OPAL Training Dataset

The future OPAL-ready USR dataset id is:

```text
rt_lnrna_sponging_construct_triage_opal_training_examples_v1
```

Use the full study prefix because this dataset may live in a mixed USR root.
Do not use generic dataset ids such as `opal_handoff_v1` or
`sponging_handoff_v1`.

OPAL readiness means this dataset has:

- one candidate universe with stable `id` values in a USR-shaped
  `records.parquet`;
- canonical base columns: `id`, `bio_type`, `sequence`, `alphabet`, and
  `length`;
- one explicit Arrow fixed-size-list vector `X` column selected by the study;
- a joined `SpongingAssayObservation` label sourced from the durable Reader
  sidecar;
- no learned `Y ~ X` campaign before that label join and fixed-size `X`
  selection are complete.

The planned label materializer is the Reader-to-Construct bridge documented in
`reader-spop-label-contract.md`. Reader owns the SPOP metric definition; the
study bridge emits the endpoint dose-mean scalar
`reader_spop_endpoint_dose_mean_v1` from pBbS2c-RFP reporter assays and routes
SPOP campaigns to OPAL as `spop_v1/spop`.

The candidate `X` columns are declared in
`../operations/contract/schemas/representation-table.schema.yaml`. Selection is
deferred until LatentDNA reviews the full construct, lnRNA-slot, RT-slot, and
slot-pair gallery views. Do not default to the largest slot-pair concat merely
because it contains more dimensions.

Pre-assay records may be OPAL-ready in shape, but OPAL `run` and `explain` are
blocked until the materialized Reader label source is joined into the training
table and satisfies OPAL's own label contract.
For shared labels, prefer `labels.source.kind: usr_sidecar` and
`writeback.prediction_records: ledger_only`.

Abundance priors are allowed as metadata or separate analysis targets. They
must not be exported as `normalized_TF_sponging_label`.

OPAL has a registered `spop_v1` objective for the one-dimensional Reader SPOP
scalar. Reader labels are now materialized as a study sidecar; this study still
must not run OPAL until a selected fixed-size `X` and those labels are both
available in the OPAL training table.
