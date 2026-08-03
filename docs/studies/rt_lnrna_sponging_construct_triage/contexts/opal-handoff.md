---
doc_id: study-rt-lnrna-sponging-construct-triage-opal-training-dataset
surface: study-context
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-08-02
---

## OPAL training dataset

The future OPAL-ready USR dataset id is:

```text
rt_lnrna_sponging_construct_triage_opal_training_examples_v1
```

Use the full study prefix because this dataset may live in a mixed USR root.
Do not use generic dataset ids such as `opal_handoff_v1` or
`sponging_handoff_v1`.

OPAL readiness requires:

- one candidate universe with stable `id` values in a USR-shaped
  `records.parquet`;
- canonical base columns: `id`, `bio_type`, `sequence`, `alphabet`, and
  `length`;
- one explicit Arrow fixed-size-list vector `X` column selected by the study;
- comparable `rt_lnrna_reporter_response_profile.v4` evidence with supported
  biological-replicate uncertainty;
- the selected 6-10 h descriptive reduction and its explicit limitations;
- a separately versioned constrained objective with validated uncertainty and
  OD interpretation;
- a versioned study label projection that maps the admitted profile to an
  ordered `Y` contract; and
- a separate OPAL objective that interprets predicted `Y` values as response
  and noncompensatory biomass-constraint channels.

The descriptive profile itself is not an OPAL label. Reader owns generic
measurements and recorded time. The RT-lnRNA study owns control assignment,
endpoint or window selection, comparability, uncertainty policy, and the later
preference objective. See `reporter-response-evidence.md`.

The candidate `X` columns are declared in
`../operations/contract/schemas/representation-table.schema.yaml`. Selection is
deferred until LatentDNA reviews the full construct, lnRNA-slot, RT-slot, and
slot-pair gallery views. Do not default to the largest slot-pair concat merely
because it contains more dimensions.

Pre-assay records may be OPAL-ready in shape, and the meta-study now recommends
6-10 h for descriptive comparison. OPAL `run` and `explain` remain blocked
until the study selects `X`, defines and validates a constrained objective, and
joins eligible profile-derived labels into the training table. For shared labels, prefer
`labels.source.kind: usr_sidecar` and
`writeback.prediction_records: ledger_only`.

Abundance priors are allowed as metadata or separate analysis targets. They
must not be exported as assay-response labels.

No objective exists yet. Do not coin an objective name or wire a generic scalar
substitute until the formula, constraint, biological-replicate unit, uncertainty method,
and validation claim are explicit.
