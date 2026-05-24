---
doc_id: study-rt-lnrna-sponging-construct-triage-source-overlays
surface: study-context
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-05-23
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

When a Khan or Crawford source row is promoted, it must move through the same
Construct and Infer path as the controls: write both `candidate__lnrna_sequence`
and `candidate__rt_cds_sequence` into
`rt_lnrna_sponging_construct_triage_construct_slot_inputs_v1`, emit the
realized 1,600 bp contexts into
`rt_lnrna_sponging_construct_triage_construct_contexts_1600bp_v1`, and attach
all six declared source sequence views before Evo2 sidecars are generated.
LatentDNA overlays then join onto those Infer-backed rows; they do not create
standalone Infer rows.

`../operations/contract/schemas/representation-table.schema.yaml` declares how
Khan and Crawford rows attach to the representation table. Khan enters as a
cross-retron RT-DNA abundance prior keyed through reference provenance.
Crawford enters through two lanes: Eco1 ncRNA abundance observations as
abundance priors, and Eco1 lnRNA/MSD design rows as sequence/design references.
Neither source may populate OPAL `Y`.

GenBank source-authority records are provenance records, not abundance priors
and not labels. They can resolve candidate sequence ids and offset checks, but
they do not create OPAL `Y` values.

### Reader SPOP

Reader retron reporter experiments are the planned source for
`SpongingAssayObservation` labels. They are not Khan/Crawford overlays and do
not replace Construct sequence authority.

The study bridge emits `reader_spop_endpoint_auc_v1` rows from
`pES-retron-*; pBbS2c-rfp` assay subjects. A Reader row can exist before a
Construct candidate row exists, but it must keep these identities separate:
`assay_subject_key`, `reader_design_id`, `proposed_candidate_id`,
`construct_candidate_id`, and `candidate_bridge_status`.

Only rows with resolved RT plus lnRNA sequence authority can join the
consolidated Construct output that feeds Infer. Unresolved Reader retron rows
remain label evidence and review overlays until their GenBank or sequence
authority is supplied.
