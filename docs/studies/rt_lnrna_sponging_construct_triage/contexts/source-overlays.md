---
doc_id: study-rt-lnrna-sponging-construct-triage-source-overlays
surface: study-context
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-05-26
---

## Source Overlay Contract

The study keeps three result layers separate:

| Layer | Meaning | May become OPAL `Y`? |
| --- | --- | --- |
| `AbundancePriorOverlay` | Literature/source msDNA or RT-DNA abundance. | No |
| `InferFeatureAlias` | Model-derived `X` for a declared construct view. | Yes, as `X` only |
| `SpongingAssayObservation` | Actual lab TF-sponging assay label. | Yes, after labels exist |

### Khan

Khan et al., "An experimental census of retrons for DNA production and genome
editing" (Nature Biotechnology, DOI `10.1038/s41587-024-02384-z`), is a
cross-retron primary literature source. It tests diverse reverse transcriptases
with cognate ncRNAs in a synthetic bacterial assay context and reports RT-DNA
production relative to Eco1 by PAGE.

The handoff has `171` terminal-keyed sequence-authority rows and `99` numeric
abundance-prior rows. Those are different layers: sequence authority tells us
which ncRNA and RT CDS can be reconstructed; the abundance overlay tells us
which reconstructed system has a source numeric RT-DNA production value. The
Construct promotion gate requires both layers, translation-exact RT CDS
validation, and fit inside the current 2,000 bp lane. It currently promotes `71`
Khan abundance-affiliated RT-lnRNA rows. The non-promoted rows are explicit
review cases: `58` sequence-authority rows fit the lane but lack affiliated
abundance, `40` exceed the current lane, and `2` lack RT CDS authority.

Use Khan `raw_value` and `normalized_value` as source-scoped RT-DNA production
fields. Use `ordinal_bin` only as secondary review metadata. These values must
not be put on one numeric scale with Crawford abundance or Reader SPOP.

### Crawford

Crawford et al., "High throughput variant libraries and machine learning yield
design rules for retron gene editors" (Nucleic Acids Research, DOI
`10.1093/nar/gkae1199`), is an Eco1-local primary literature source. It
characterizes how retron Eco1 ncRNA/lnRNA/MSD sequence changes affect msDNA
abundance in a controlled variant-library setting.

The current inventory has `2,578` design-reference rows and `4,174` abundance
observation rows. Those are different row grains. Design references carry
sequence and MSD/MSR decomposition provenance; abundance observations carry
source numeric msDNA abundance evidence, including repeated or duplicate
source-observation rows that should not be collapsed away. Preserve raw numeric
fields. Do not average Crawford and Khan into one abundance target.

Promoted Crawford rows are source lnRNA sequences projected into the dnadesign
dual-cassette construct. They must have an affiliated abundance-observation row;
design-reference-only sequences are noted as source provenance but do not move
forward as Construct candidates. A promoted sequence may also have one or more
design-reference rows, but abundance affiliation is the promotion gate. These
rows are not asserted to recreate the exact Crawford synthetic expression
context, and their A1/A2 extension geometry is not assumed to match the
dnadesign A1/A2=20 convention.

### Compiler-Generated MSD

Compiler-generated MSD rows are study-owned sequence/design references derived
from bounded Retron MSD primitive combinations. The study uses `YIU` as the
canonical acronym for the YIU-compatible cloning method. In this ontology, the
candidate MSD geometry is composed from a Snapback cap primitive and a
scar-nick stem-base primitive, then compiled into an MSD sequence, reverse
complemented into the larger long non-coding RNA, and finally projected through
Construct. The current fixture pool resolves five DE033 Snapback cap ranks and
sixteen scar-nick TetO stem-base ranks, yielding the full 5 x 16 = 80 design
space for the TetO payload without hand-enumerating each combination.

Compiler-generated rows are Crawford-like only in the sense that they
contribute lnRNA variant sequence authority. They are not literature abundance
priors, not OPAL `Y`, and not a pre-Infer concat lane.

The promotion gate compiles a pure MSD unit through the Retron compiler,
records the payload, cap rank, stem-base rank, Snapback topology, scar-nick
route, nick orientation, and nickase provenance, then patches the template
lnRNA only when the insert equals the reverse complement of the 5-prime-to-3-prime
MSD product. The template must contain exactly one reverse-complemented
template MSD span and the declared 5-prime and 3-prime flanks. Duplicate
generated lnRNA sequences and Construct window violations fail before
USR/Construct materialization.

### Join Rule

Source rows can become overlays or provenance records. They are not construct
views and are not lab TF-sponging labels. A source row becomes a construct subject row
only after the study names a construct-compatible RT plus lnRNA pairing and it
passes representability checks.

When a Khan, Crawford, GenBank, or compiler-generated MSD source row is
promoted, it must move through the same Construct and Infer path as every other
first-class Construct subject: write both
`construct_subject__lnrna_sequence` and `construct_subject__rt_cds_sequence` into
`rt_lnrna_sponging_construct_triage_construct_slot_inputs_v1`, emit the
realized 2,000 bp contexts into
`rt_lnrna_sponging_construct_triage_construct_contexts_2000bp_v1`, and attach
all six declared source sequence views before Evo2 sidecars are generated.
LatentDNA overlays then join onto those Infer-backed rows; they do not create
standalone Infer rows.

`../operations/contract/schemas/representation-table.schema.yaml` declares how
Khan and Crawford rows attach to the representation table. Khan enters as a
cross-retron RT-DNA abundance prior keyed through reference provenance.
Crawford enters through two lanes: Eco1 ncRNA abundance observations as
source-scoped abundance priors, and Eco1 lnRNA/MSD design rows as
sequence/design references. Neither source may populate OPAL `Y`, and neither
source's numeric field may be treated as numerically comparable to the other.

The current source-promotion resolver treats Crawford as an Eco1-local lnRNA
source and pairs abundance-affiliated source sequences with fixed WT Eco1 RT. It
promotes a Crawford sequence only when an abundance row exists, the DNA sequence
passes Eco1 forward k-mer orientation QC, and the sequence does not look
reverse-complemented. Exact declared MSD substring and short flank matches are
preserved as QC annotations because Crawford variants may intentionally alter
those regions; missing anchors do not make an abundance-bearing source sequence
inherently invalid. Khan rows are ingested through the terminal-keyed
sequence-authority table, but they are not
Construct-promoted unless an explicit source RT CDS DNA sequence exists. RT
accessions and RT-DNA product sequences are provenance, not RT CDS authority.
Both lanes use the Construct projection manifest's lnRNA-centered window
geometry for promotion; there is no independent combined-length shortcut. Khan
rows without affiliated abundance priors are retained as sequence authority and
review provenance, not Construct subjects.

GenBank source-authority records are provenance records, not abundance priors
and not labels. They resolve candidate sequence ids, parsed slot spans, and
offset checks, but they do not create OPAL `Y` values. All representable
GenBank catalog rows use the same source-authority and Construct projection
path.

### Reader SPOP

Reader retron reporter experiments are the planned source for
`SpongingAssayObservation` labels. They are not Khan/Crawford overlays and do
not replace Construct sequence authority.

Reader owns the SPOP metric definition in the sibling source-of-truth document
`reader/docs/lib/spop_endpoint_in_reader.md` and public scoring API
`reader.domains.plate_reader.analysis.spop.score_spop_endpoint`. The RT-lnRNA
study owns only the Construct bridge. The study bridge emits
`reader_spop_endpoint_dose_mean_v1` rows from
`pES-retron-*; pBbS2c-rfp` assay subjects. A Reader row can exist before a
Construct subject row exists, but it must keep these identities separate:
`assay_subject_key`, `reader_design_id`, `proposed_construct_subject_id`,
`construct_subject_id`, and `construct_subject_bridge_status`.

Only rows with resolved RT plus lnRNA sequence authority can join the
consolidated Construct output that feeds Infer. Unresolved Reader retron rows
remain label evidence and review overlays until their GenBank or sequence
authority is supplied.

Reader SPOP is the materialized lab TF-sponging scalar for LatentDNA GenBank
overlays. Its numeric scope is
`reader_experiment_normalized_tf_sponging`, separate from Khan RT-DNA abundance
and Crawford Eco1 msDNA abundance. Numeric scoring is delegated to Reader's
public `score_spop_endpoint` API. It may be used as a categorical hue, an
ordinal audit axis, or OPAL `Y` only after the row has a resolved Construct
subject bridge and the materialized `SpongingAssayObservation` contract is
written.
