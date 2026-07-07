---
doc_id: study-rt-lnrna-sponging-construct-triage-reader-spop-label-contract
surface: study-context
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-05-31
---

## Reader SPOP Construct Bridge

Reader is the source-of-truth owner for SPOP assay metric semantics. The
Reader-owned reference document is `reader/docs/lib/spop_endpoint_in_reader.md`
in the sibling Reader repository, and the public scoring surface is
`reader.domains.plate_reader.analysis.spop.score_spop_endpoint`.

This study document is narrower: it describes how the RT-lnRNA study bridges
Reader SPOP observations onto Construct subject identity. The study-owned
planner in
`src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_plan.py`
converts selected sibling Reader pES-retron plus pBbS2c-RFP experiments into a
single scalar:

```text
reader_spop_endpoint_dose_mean_v1
```

The scalar is endpoint RFP/OD600 derepression under nonzero IPTG, normalized
within each Reader experiment by the zero-inducer baseline and the aTc positive
control. Positive control means positive aTc and zero IPTG; the actual aTc dose
is recorded per observation because historical Reader retron benchmarks include
both 20 nM and 200 nM aTc positive controls. This is an endpoint dose-ladder mean, not an AUC.

The bridge delegates all numeric scoring to Reader's public
`score_spop_endpoint` API. It must not duplicate SPOP math in dnadesign. The
study bridge only parses Reader artifacts, resolves Construct subject identity,
passes dose values to Reader, and persists the Reader-owned scorer output plus
QC metadata.

SPOP expands to `sponging_percent_of_positive`. The metric scope is
`reader_experiment_normalized_tf_sponging`. It is a source-scoped Reader assay
scalar, not a Khan RT-DNA abundance value, not a Crawford Eco1 msDNA abundance
value, and not a Construct sequence property. Downstream audits may derive
ordinal bins or categorical hues from SPOP, but those bins are metadata views
over this Reader metric and must preserve the underlying numeric value.

### Identity Boundary

Do not collapse assay identity into construct identity.

| Field | Meaning |
| --- | --- |
| `assay_subject_key` | Reader-local subject such as `retron26` or `retron176`. |
| `reader_design_id` | Raw Reader design id, for example `pES-retron-26; pBbS2c-rfp`. |
| `proposed_construct_subject_id` | Study proposal for the eventual RT plus lnRNA construct subject row. |
| `construct_subject_id` | Real Construct-backed subject id; null until sequence authority exists. |
| `construct_subject_bridge_status` | Whether this row has resolved Construct sequence authority. |

Reader retron numbers resolve through the variant GenBank catalog when explicit
RT plus lnRNA sequence authority exists and the construct subject is
Construct-representable. Rows without catalog authority may carry assay labels,
but they must remain `missing_construct_sequence_authority` until promoted
through Construct.

As of 2026-07-07, the live Reader bridge resolves the observed catalog-backed
retron rows to Construct subject ids, including retron47/retron48, retron49-56,
retron170-175, retron177-186, retron195-200, retron26, retron43, and retron180.
It materializes 50 Reader SPOP observations across 36 candidate summaries for
LatentDNA overlays. The 2025-11-05 RT-variant experiment is a single-point
mid-log read, not a time course; the Reader artifact stores row time as 0 h, but
the study planner records the endpoint as approximately 10 h after seeding and
adds `single_point_endpoint_time_override`. The 2026-05-07 retron176 wells are
omitted from labels because the plate map carried retron176 but no actual strain
was present in those wells.

### Reader Evidence

The study bridge must resolve Reader artifacts through
`outputs/manifests/records.json`, using latest record
`ratio_reporter_normalizer/df`. Direct path scraping is not allowed because the
label must carry a stable `reader_artifact_ref`,
`reader_artifact_record_id`, and `reader_artifact_content_digest`.

The reporter plasmid metadata is `pBbS2c-RFP`; Reader design ids commonly use
`pBbS2c-rfp` and are parsed case-insensitively.

### OPAL Handoff

OPAL receives this as a one-dimensional SPOP label through `spop_v1/spop`.
`spop_v1` is the modular OPAL objective for Reader SPOP scalar campaigns. Use
`scalar_identity_v1/scalar` only for generic scalar smoke tests, not for the
RT-lnRNA SPOP campaign contract.

The future OPAL training table may include rows only when both conditions hold:

1. one selected fixed-size vector `X` has been chosen after LatentDNA review;
2. the row has a real Reader-derived SPOP scalar and any row used as a Construct
   candidate has resolved sequence authority.

Before OPAL handoff, the materialized candidate summary can be joined as a
LatentDNA overlay or ordinal audit axis when
`construct_subject_bridge_status` is `resolved_construct_sequence_authority`.
Rows without that bridge stay as assay evidence only and must not be silently
promoted into Infer-backed training rows.
