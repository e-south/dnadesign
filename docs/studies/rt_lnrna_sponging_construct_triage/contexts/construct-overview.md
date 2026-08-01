---
doc_id: study-rt-lnrna-sponging-construct-triage-construct-overview
surface: study-context
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-07-29
---

## RT-lnRNA Construct Overview

This study turns paired reverse-transcriptase CDS and long non-coding RNA
subjects into fixed 2,000 bp dual-cassette Construct contexts. Each subject has
one `lnrna` slot, one `rt_cds` slot, and one semantic
`construct_subject__id`. Construct writes one forward context row, one
reverse-complement context row, and six declared sequence-view rows per subject.

The Construct input dataset is
`rt_lnrna_sponging_construct_triage_construct_slot_inputs_v1`. Its USR row id is
a carrier. Biological identity and sequence authority live in
`construct_subject__id`, `construct_subject__lnrna_sequence`, and
`construct_subject__rt_cds_sequence`.

The Construct output dataset is
`rt_lnrna_sponging_construct_triage_construct_contexts_2000bp_v1`. It stores the
realized context sequence, `construct__slots` spans for the exact lnRNA and RT
CDS placements, and a `construct_subject` bridge back to the input subject.

### Source Families

| Source | Primary meaning | Construct promotion |
| --- | --- | --- |
| RT-lnRNA subject bindings | Study-owned composition of provider-owned RT parts, lnRNA parts, exact Reader aliases, and optional hairpin structure references. The binding set contains 49 logical subjects. | The unified selector accepts biological subjects only through the resolved binding API. It validates component digests before writing Construct inputs. |
| GenBank catalog | Study-owned sequence and placement provenance for 46 historical RT-lnRNA compositions, including whole-plasmid and lnRNA-only records. | It supplies component bytes and Construct geometry. It does not independently mint subjects; the catalog-only audit materializer requires exact coverage by the same subject-binding resolver. |
| Crawford et al. | Primary literature source: Crawford et al., "High throughput variant libraries and machine learning yield design rules for retron gene editors", NAR, DOI `10.1093/nar/gkae1199`. The source characterizes Eco1 retron ncRNA/lnRNA/MSD variants and reports msDNA abundance scores relative to mean wild type. | The dnadesign-data handoff has 4,174 abundance-observation rows and 2,578 design-reference rows. These are different row grains: abundance observations are numeric assay evidence, while design references describe sequence/design provenance. The resolver promotes 4,148 abundance-affiliated lnRNA sequences with fixed Eco1 WT RT. The remaining design-reference-only sequences stay as provenance and issue records. |
| Khan et al. | Primary literature source: Khan et al., "An experimental census of retrons for DNA production and genome editing", Nature Biotechnology, DOI `10.1038/s41587-024-02384-z`. The source places diverse retron RT plus cognate ncRNA systems in a synthetic assay context and reports RT-DNA production relative to Eco1 by PAGE. | The dnadesign-data handoff has 171 terminal-keyed sequence-authority rows and 99 numeric abundance-prior rows. The resolver promotes rows with explicit ncRNA sequence, translation-exact RT CDS authority, affiliated abundance prior, and fit inside the current 2,000 bp lane. That yields 71 Khan Construct subjects; 58 fit the lane but lack affiliated abundance, 40 exceed the lane, and 2 lack RT CDS authority. |
| Study MSD compiler pool | Study-owned YIU-compatible lnRNA/MSD design pool. The compiler combines Cruncher-derived Snapback cap primitives and scar-nick TetO stem-base primitives into putative MSD subcomponents before the larger lnRNA is projected through Construct. | Compiler promotion is off by default. The checked-in operational recipe opts in only after the hairpin owner validates the configured primitives. |
| Permuter RT-CDS DMS | In silico RT-CDS variants generated through the public `dnadesign.permuter` API. | The current plan generates 6,080 single-codon RT-CDS DMS subjects from Eco1 WT RT and keeps Permuter provenance in the construct-subject overlay. |
| Reader-backed reporter response | Descriptive assay profiles derived by the RT-lnRNA study from generic Reader measurement records and explicit scientific policy. | The profile is assay evidence, not sequence authority or an optimization label. It joins only after a Reader assay subject resolves to a study subject. |

The minimal public materialization call selects only registered RT-lnRNA
subjects. Crawford/Khan source promotions, compiler-generated lnRNA variants,
and RT-CDS DMS are independent opt-in lanes. The broader checked-in campaign
recipe names every enabled lane explicitly.

The lanes remain separate and report their own counts. No aggregate count or
catalog row is treated as a second source of biological subject identity.

### Source Numerics

Crawford, Khan, and reporter-response numeric values are source-scoped. They
are not one shared abundance scale.

- Crawford `raw_value`/`normalized_value` describes Eco1-local msDNA abundance
  relative to mean wild type in the Crawford source.
- Khan `raw_value`/`normalized_value` describes RT-DNA production relative to
  Eco1 in the Khan source.
- Reporter-response profiles preserve dose-wise response and relative OD after
  the study bridge resolves an exact subject and explicit control policy.

Ordinal bins and categorical hues are review metadata. They can color
LatentDNA/OPAL audits, but they do not replace the source numeric fields and do
not become OPAL `Y` unless the target contract says so.

### Infer Views

Infer consumes the six source views by explicit `view_name`:

| View | Meaning | Pooling |
| --- | --- | --- |
| `dual_cassette_2000bp_seq_mean` | Forward full 2,000 bp context. | `seq_mean` over the whole context. |
| `dual_cassette_2000bp_reverse_complement_seq_mean` | Reverse-complement full 2,000 bp context. | `seq_mean` over the whole RC context. |
| `lnrna_fixed_384bp_window_in_construct_anchor_mean` | Forward full context with a fixed 384 bp lnRNA-centered pooling window. | `anchor_mean` using sequence-view bounds. |
| `lnrna_fixed_384bp_window_in_construct_reverse_complement_anchor_mean` | RC full context with the orientation-aware 384 bp lnRNA pooling window. | `anchor_mean` using sequence-view bounds. |
| `rt_cds_fixed_1600bp_window_in_construct_anchor_mean` | Forward full context with a fixed 1,600 bp RT-CDS-centered pooling window. | `anchor_mean` using sequence-view bounds. |
| `rt_cds_fixed_1600bp_window_in_construct_reverse_complement_anchor_mean` | RC full context with the orientation-aware 1,600 bp RT CDS pooling window. | `anchor_mean` using sequence-view bounds. |

The fixed windows normalize pooling length while preserving anchor placement.
Construct centers each window on the named slot when possible and shifts it
inside the 2,000 bp context when the slot is near a context boundary. The exact
biological slot span remains in `construct__slots`; the sequence-view
`anchor_start_0` and `anchor_end_0` fields carry the fixed pooling window. A
slot that cannot fit inside its fixed window fails before Infer.

For each view, Infer writes Evo2 7B sidecars for the block 26 intermediate
embedding, output-layer mean, `log_likelihood__total`, and
`log_likelihood__mean_per_token`. LatentDNA derives bidirectional concat views
only after those sidecars exist.
