---
doc_id: study-rt-lnrna-sponging-construct-triage-source-overlays
surface: study-context
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-05-22
---

## Source Overlay Contract

The study keeps three result layers separate:

| Layer | Meaning | May become OPAL `Y`? |
| --- | --- | --- |
| `AbundancePriorOverlay` | Literature/source msDNA or RT-DNA abundance. | No |
| `InferFeatureAlias` | Model-derived `X` for a declared construct view. | Yes, as `X` only |
| `SpongingAssayObservation` | Actual lab TF-sponging assay label. | Yes, after labels exist |

### Khan

Khan source rows are cross-retron RT-DNA abundance priors. The overlay path is
listed in `../record/datasets.yaml` with `99` abundance-prior rows. Use
`raw_value` and `normalized_value` as primary numeric fields; use
`ordinal_bin` only as secondary analysis metadata.

### Crawford

Crawford source rows are Eco1-local lnRNA/MSD references and abundance
observations. The current inventory has `2578` design reference rows and
`4174` abundance observation rows. Preserve raw numeric fields. Do not average
Crawford and Khan into one abundance target.

### Join Rule

Source rows can become overlays or provenance records. They are not construct
views and are not lab TF-sponging labels. A source row becomes a candidate row
only after the study names a construct-compatible RT plus lnRNA pairing and it
passes representability checks.

GenBank source-authority records are provenance records, not abundance priors
and not labels. They can resolve candidate sequence ids and offset checks, but
they do not create OPAL `Y` values.
