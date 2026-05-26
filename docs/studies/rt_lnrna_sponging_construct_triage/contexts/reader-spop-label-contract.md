---
doc_id: study-rt-lnrna-sponging-construct-triage-reader-spop-label-contract
surface: study-context
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-05-24
---

## Reader SPOP Label Contract

Reader is the evidence owner for TF-sponging assay labels. The study-owned
planner in
`src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_plan.py`
converts selected sibling Reader pES-retron plus pBbS2c-RFP experiments into a
single scalar:

```text
reader_spop_endpoint_auc_v1
```

The scalar is endpoint RFP/OD600 derepression under nonzero IPTG, normalized
within each Reader experiment by the zero-inducer baseline and the aTc positive
control:

```text
y(dose) = (Z_IPTG(dose) - Z_0) / (Z_aTc - Z_0)
score = mean(max(0, y(dose))) * ((1 - lambda) + lambda * mean(viability))
```

`lambda` defaults to `0.5`. Viability is the one-sided OD600 ratio relative to
the zero-IPTG baseline. Raw dose-level `y` values are retained for QC even when
the primary scalar clips negative values at zero.

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

As of 2026-05-24, the live Reader dry run resolves the observed catalog-backed
retron rows to Construct subject ids, including retron47/retron48 and
retron49-56. The 2025-11-05 RT-variant experiment is a single-point mid-log
read, not a time course; the Reader artifact stores row time as 0 h, but the
study planner records the endpoint as approximately 10 h after seeding and
adds `single_point_endpoint_time_override`. The 2026-05-07 retron176 wells are
omitted from labels because the plate map carried retron176 but no actual strain
was present in those wells.

### Reader Evidence

The planner must resolve Reader artifacts through
`outputs/manifests/records.json`, using latest record
`ratio_reporter_normalizer/df`. Direct path scraping is not allowed because the
label must carry a stable `reader_artifact_ref`,
`reader_artifact_record_id`, and `reader_artifact_content_digest`.

The reporter plasmid metadata is `pBbS2c-RFP`; Reader design ids commonly use
`pBbS2c-rfp` and are parsed case-insensitively.

### OPAL Handoff

OPAL receives this as a one-dimensional scalar label through
`scalar_identity_v1/scalar`.

OPAL must not run OPAL `spop_v1` for this study. The OPAL `spop` note is a
historical draft objective specification, not a registered runtime objective
for the current RT-lnRNA study.

The future OPAL training table may include rows only when both conditions hold:

1. one selected fixed-size vector `X` has been chosen after LatentDNA review;
2. the row has a real Reader-derived SPOP scalar and any row used as a Construct
   candidate has resolved sequence authority.
