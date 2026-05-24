---
doc_id: study-rt-lnrna-sponging-construct-triage-opal-training-dataset
surface: study-context
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-05-23
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
- a future label slot for `SpongingAssayObservation`;
- no learned `Y ~ X` campaign before real labels exist.

The planned label materializer is the Reader bridge documented in
`reader-spop-label-contract.md`. It emits the scalar
`reader_spop_endpoint_auc_v1` from pBbS2c-RFP reporter assays and routes it to
OPAL as `scalar_identity_v1/scalar`.

The candidate `X` columns are declared in
`../operations/contract/schemas/representation-table.schema.yaml`. Selection is
deferred until LatentDNA reviews the full construct, lnRNA-slot, RT-slot, and
slot-pair gallery views. Do not default to the largest slot-pair concat merely
because it contains more dimensions.

Pre-assay records may be OPAL-ready in shape, but OPAL `run` and `explain` are
blocked until a configured label source satisfies OPAL's own label contract.
For shared labels, prefer `labels.source.kind: usr_sidecar` and
`writeback.prediction_records: ledger_only`.

Abundance priors are allowed as metadata or separate analysis targets. They
must not be exported as `normalized_TF_sponging_label`.

The historical OPAL `spop` objective note remains a draft specification. This
study should not call an OPAL `spop_v1` objective unless that plugin is
implemented and registered separately; the pragmatic handoff is a
one-dimensional Reader scalar.
