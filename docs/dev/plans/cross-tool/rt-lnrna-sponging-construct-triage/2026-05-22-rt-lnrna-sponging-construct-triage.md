## RT-lnRNA Sponging Construct Triage

**Status:** proposed development specification
**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-22
**Study id:** `rt_lnrna_sponging_construct_triage`
**Primary study surface:** planned

This proposal defines a contract-first implementation path for a synthetic
RT-lnRNA expression construct representation workbench. It is a cross-tool
proposal because the durable boundary crosses study records, Construct, USR,
Infer, Permuter, OPAL, and source-backed handoffs from `dnadesign-data`.

### 0. Document Framing And Authority

**Audience:** maintainers of the planned study surface plus Construct, USR,
Infer, Permuter, OPAL, and `dnadesign-data` handoff maintainers.

**Decision outcome:** approve a narrow Phase 0/Phase 1 implementation contract
for a paired synthetic RT-lnRNA construct workbench, not a generic retron atlas
and not an RT-only model-catalog project.

**Authoritative scope corrections for this draft:**

- The core unit is `SyntheticRtLnrnaSpongingConstruct`.
- Eco1 WT RT with Eco1-derived retron26/retron43 lnRNA designs is the v1
  anchor profile, not the reusable ontology.
- `perturbation_class`, `cloning_constraint_set`, and
  `representation_result` are not core objects.
- Khan and Crawford are `AbundancePriorOverlay` regimes; they are not
  TF-sponging labels, construct views, or Infer outputs.
- OPAL readiness means stable candidate ids, explicit `X` feature aliases, and
  a future label slot. It does not mean learning before
  `SpongingAssayObservation` rows exist.

### 1. Problem Statement And North Star

The central semantic unit is:

```text
Synthetic RT-lnRNA expression construct for programmable multicopy ssDNA sponging
```

A row means one lnRNA/msr-msd-payload cassette plus one RT coding sequence
cassette, placed into a fixed synthetic dual-expression plasmid context with
the lnRNA cassette upstream and the RT cassette following downstream. The row
is the paired construct, not an RT-only catalog row, not an msd-only design
row, and not a native retron locus.

The biological goal is to compare paired RT + lnRNA/msr-msd-payload systems in
a fixed synthetic vector context so construct-context Evo2 embeddings can be
tested for whether they contain enough multicopy ssDNA / RT-DNA abundance
structure to support limited, interpretable wet-lab triage.

The v1 anchor profile is Eco1 WT RT with Eco1-derived lnRNA designs:

```yaml
study_id: rt_lnrna_sponging_construct_triage
description: Synthetic RT-lnRNA expression construct triage for programmable multicopy ssDNA sponging.
construct_contract: dual_cassette_rt_lnrna_expression_v1
representation_contract: dual_cassette_construct_context_embedding_v1
payload_program_id: tetO_sponging_v1
anchor_working: eco1_wt_rt__eco1_derived_lnrna_retron26
anchor_failed: eco1_wt_rt__eco1_derived_lnrna_retron43
reference_overlays:
  - khan_cross_retron_rt_dna_abundance_v1
  - crawford_eco1_lnrna_msd_abundance_v1
```

Eco1 is the first anchor/profile, not the ontology. The reusable category is
the synthetic RT-lnRNA expression construct. Before lab TF-sponging labels
exist, embeddings are exploratory evidence only. They may help choose a small
batch if working/failed controls separate and obvious confounds do not explain
the geometry. They must not be presented as direct functional predictions.

### 2. Existing Capabilities And Architectural Boundaries

The implementation must preserve current repo boundaries:

- Study records own biological framing, candidate rationale, route maps, and
  phase status.
- Construct owns standardized construct sequence realization and emitted
  coordinate metadata. Downstream tools must not duplicate construct assembly
  logic.
- USR owns durable sequence records, identities, overlays, sequence views, and
  additive sidecars.
- Infer consumes explicit sequence views and writes feature aliases/sidecars.
  It must not invent constructs, missing windows, or source-derived sequence
  products.
- Permuter is downstream of selected references and mutation grammars. It can
  generate controlled RT codon/AA variants or other selected variants, but it
  must not own external source discovery.
- OPAL starts after one OPAL-ready candidate table with one explicit `X` column
  exists. It should not learn before real `SpongingAssayObservation` labels
  exist.
- `dnadesign-data` provides source-backed reference rows, numeric abundance
  priors, and provenance. It does not provide construct views, Infer results,
  or TF-sponging assay labels.

The current retron hairpin study is compiler-oriented. It can supply MSD design
primitives, study rationale, Snapback/scar-nick provenance, and existing
retron26/retron43-related design records. It should not become an RT catalog,
a database crawler, or a generic protein modeling surface.

If a future catalog component is promoted into `dnadesign`, it must become an
explicit top-level boundary or an explicit shared contract. It must not be a
hidden helper inside Infer, Permuter, Construct, Cruncher, or the existing
retron hairpin study.

#### Repo Surface Alignment

This plan should reuse current contracts rather than invent parallel names:

| Surface | Current contract expectation | Requirement for this study |
| --- | --- | --- |
| Construct | One construct job realizes one template against one input selection and emits `construct__*` lineage plus optional forward/reverse-complement `realized_context` sequence views. Multiple named slots may be assembled from one candidate row inside that template; matrix studies across templates remain multiple project entries. | Use the public `construct_multi_slot_assembly_v1` path for lnRNA and RT placement. Do not precompose lnRNA and RT into one hidden runtime anchor. |
| USR | Base `records.parquet` uses `id`, `bio_type`, `sequence`, `alphabet`, `length`, `source`, and `created_at`; richer semantics live in namespaced columns and `_views/sequence_views.parquet`. Product kinds are generic lineage terms such as `construct_insert` and `realized_context`. | Do not add RT/lnRNA-specific product kinds. Use base USR records for sequences, sequence views for emitted construct contexts, and `_views/view_semantics.parquet` for study membership, source family, and role tags. |
| Infer | Evo2 feature bundles consume explicit sequence views and write aliases/payloads under `_derived/infer/`. `anchor_mean` sends the full emitted context through Evo2 and pools the declared emitted-orientation span. | Construct or USR must provide every forward/RC sequence view and pooling span before Infer runs. Infer must not reverse-complement, window, pad, or infer biological spans implicitly. |
| LatentDNA | Workspaces materialize source-backed vector views, derived views, metrics, plots, notebooks, and review surfaces from explicit config. `infer_feature_sidecar` sources expose Infer aliases joined to vectors and sequence-view metadata. | LatentDNA may own representation review, block-normalized forward/RC concatenation, landmark/anchor metrics, projections, and notebooks. It is not the source of truth for construct projection, sequence identity, or OPAL campaign labels. |
| OPAL | OPAL starts from a candidate `records.parquet` with one explicit fixed-length `X` column. For shared labels, `labels.source.kind: usr_sidecar` plus `writeback.prediction_records: ledger_only` keeps assay truth out of campaign ledgers. | OPAL sees one selected `X` column after Infer/LatentDNA export. It does not select the representation view, learn from abundance priors as `Y`, or mutate upstream candidate definitions. |
| Ops/studies | A checked-in study owns `README.md`, `record/`, `operations/ops.study.yaml`, optional route maps, contexts, and workbench artifacts. Ops status/preflight surfaces are added only when real providers exist. | Phase 0 can create study records and planned contracts. Do not register `studies.rt-lnrna-sponging-construct-triage.status` until a concrete provider exists. |

Phase 0 should pin these existing contract facts before writing any candidate
or generated artifact:

- Construct is one-template-per-job and one input selection per job, with
  public named-slot assembly inside that selected candidate row. The lnRNA and
  RT CDS are separate Construct parts/slots under
  `construct_multi_slot_assembly_v1`.
- USR sequence-view identity is narrow: `product_kind`, `orientation`, parent
  lineage, optional source interval, optional anchor bounds, template ids, and
  `analysis_only`. Study roles and source regimes belong in
  `_views/view_semantics.parquet` or study tables.
- Infer sequence-view bundles accept declared `sequence_view_inputs[]`,
  `seq_mean`, `anchor_mean`, and `core60_mean`. `anchor_mean` requires bounds
  from the sequence view or construct overlay and still runs Evo2 over the full
  emitted sequence.
- LatentDNA can consume `infer_feature_sidecar` and
  `infer_feature_scalar_sidecar` sources, and can build
  `block_normalized_concatenate` views. It should not become the candidate or
  label authority.
- OPAL's runtime X contract is a finite Arrow fixed-size-list vector column in
  `records.parquet`, not a notebook artifact, UMAP, ragged list, JSON string,
  or abundance-prior column.

### 3. Target Study And Tool Surface

The first implementation should create a checked-in study record only when
Phase 0 starts. The planned study shape is:

```text
docs/studies/rt_lnrna_sponging_construct_triage/
  README.md
  record/
    campaign.yaml        # optional; required if LatentDNA study_binding or an OPAL campaign manifest uses it
    datasets.yaml
    status.md
  operations/
    ops.study.yaml
    runtime/
      command-groups/pipeline.yaml
  routes/
    README.md
  contexts/
    construct-contract.md
    representation-contract.md
    source-overlays.md
    opal-handoff.md
  workbench/
    ontology/
    design_sets/
    provenance/
```

Study implementation helpers, if needed, should live under:

```text
src/dnadesign/studies/studies/rt_lnrna_sponging_construct_triage/
```

Those helpers should stay narrow: schema parsing, candidate-table validation,
study-owned analysis metrics, and study-owned batch handoff. Reusable construct
projection belongs in Construct or a neutral contract. Reusable feature
extraction belongs in Infer. Reusable candidate table active learning belongs
in OPAL after labels exist.

The study should also keep a short route map once Phase 0 starts:

| Request | First owner surface |
| --- | --- |
| "What is the biological scope?" | `docs/studies/rt_lnrna_sponging_construct_triage/routes/README.md` |
| "What sequence is embedded?" | `contexts/construct-contract.md` and `contexts/representation-contract.md` |
| "Which source overlays are allowed?" | `contexts/source-overlays.md` plus `record/datasets.yaml` |
| "Is the representation ready for triage?" | LatentDNA review context plus study-owned analysis report |
| "Can OPAL start?" | `contexts/opal-handoff.md` and OPAL config validation, after real labels exist |

### 4. Core Ontology And Schema Definitions

The lean core ontology is:

| Object | Purpose | Owner |
| --- | --- | --- |
| `SyntheticRtLnrnaSpongingConstruct` | One lnRNA/msr-msd-payload cassette followed by one RT CDS cassette in the fixed synthetic dual-expression context. | Study candidate table |
| `RtCds` | DNA coding sequence for the reverse transcriptase used in a candidate row. | USR/source handoff |
| `RtProteinProvenance` | Source, accession, variant, mutation, and literature/support metadata for the RT protein encoded by `RtCds`. | Source overlay / study |
| `LnrnaSequence` | The lnRNA cassette sequence used in the construct; in construct context this is represented as DNA encoding the transcript. | USR/source handoff |
| `LnrnaProvenance` | Source, design id, derivation, source row, and literature/support metadata for the lnRNA. | Source overlay / study |
| `MsdDesignSpec` | Study-owned design metadata for the MSD/hairpin/payload portion, including compiler source and feasibility status. | Study/compiler |
| `DualExpressionConstruct` | The concrete synthetic plasmid-region construct instance produced by placing lnRNA and RT anchors into the contract template. | Construct |
| `ConstructContextView` | A declared sequence product/view to embed, with coordinates, length policy, orientation, pooling intent, and failure behavior. | Construct / USR |
| `SpongingAssayObservation` | Future lab TF-sponging label rows with assay metadata and normalization. | Study / OPAL label source |
| `InferFeatureAlias` | Pointer to model-derived feature aliases, vector sidecars, scalar sidecars, and export-ready `X` columns. | Infer |
| `AbundancePriorOverlay` | Literature/source abundance labels used as priors or reference overlays, not sponging labels. | dnadesign-data / USR overlay |

The following must not be core ontology objects:

- `perturbation_class`
- `cloning_constraint_set`
- `representation_result`

Replacement semantics:

- Use optional `variant_derivation` metadata to explain how a candidate was
  generated, for example `p4_stem_extension`, `snapback_cap_swap`,
  `scar_nick_base_pair_swap`, `rt_catalytic_dead_control`, or
  `high_producer_reference_import`.
- Put cloning and design feasibility details inside `MsdDesignSpec`, because
  those constraints apply to MSD/hairpin subcomponents rather than the whole
  RT-lnRNA construct.
- Use `ConstructContextView` for what sequence is embedded, and
  `InferFeatureAlias` for where computed model outputs live.

For the v1 Eco1 anchors, normalize the lab system as:

```yaml
working_anchor:
  rt_source: Eco1
  rt_variant: WT
  lnrna_source: Eco1-derived
  lnrna_design_id: retron26
  payload_program_id: tetO_sponging_v1

failed_anchor:
  rt_source: Eco1
  rt_variant: WT
  lnrna_source: Eco1-derived
  lnrna_design_id: retron43
  variant_derivation: p4_stem_extension
  payload_program_id: tetO_sponging_v1
```

Do not collapse these to the label "Eco1." The biological row is Eco1 WT RT
plus a specific Eco1-derived lnRNA design.

### 5. Construct Contract

The construct contract is:

```yaml
construct_contract: dual_cassette_rt_lnrna_expression_v1
```

It represents a fixed synthetic plasmid regional context:

```text
[plasmid prefix context]
[promoter_lnrna][lnRNA/msr-msd-payload][terminator_lnrna]
[interstitial region]
[promoter_RT][RBS_RT][RT_CDS][terminator_RT]
[plasmid suffix context]
```

Constants stay fixed within a construct contract version. Variable anchors are
primarily:

- `lnRNA/msr-msd-payload`
- `RT_CDS`

Conceptually the construct has two biological anchors, but the current
Construct runtime must expose that expressivity directly rather than using a
hidden precomposed anchor. The v1 strategy is
`construct_multi_slot_assembly_v1`: one candidate row binds separate named
slots for the lnRNA cassette and RT CDS into one fixed expression-vector
template. Construct owns generic slot assembly, placement guards, span emission,
reverse-complement output variants, and runtime lineage. The study owns RT,
lnRNA, payload, source-overlay, and representation semantics.

`SyntheticRtLnrnaSpongingConstruct` remains the study row. Construct slots are
generic runtime parts, not RT/lnRNA ontology records; those meanings are
preserved through candidate fields, source authority, and view semantics.

For v1, Eco1 WT RT is fixed for the main search. The primary variants are
Eco1-derived lnRNA/MSD designs, especially retron26, retron43, weak/rescue
anchors when available, and cloning-feasible variants from existing compiler
primitives.

The construct contract must define:

| Field | Required behavior |
| --- | --- |
| `construct_contract` | Literal `dual_cassette_rt_lnrna_expression_v1`. |
| `construct_template_id` | Stable id for the fixed plasmid regional template. |
| `plasmid_context_source_id` | Real source of prefix/suffix sequence; no arbitrary filler. |
| `lnrna_promoter_id`, `lnrna_terminator_id` | Fixed lnRNA cassette constants. |
| `rt_promoter_id`, `rt_rbs_id`, `rt_terminator_id` | Fixed RT cassette constants. |
| `interstitial_region_id` | Fixed region between cassettes. |
| `lnrna_anchor_id` | Candidate-specific lnRNA cassette sequence id. |
| `rt_anchor_id` | Candidate-specific RT CDS sequence id. |
| `construct_slots` | Named Construct slots with sequence fields, placement guards, template spans, and required-slot status. |
| `anchor_spans` | 0-based half-open spans in the emitted construct sequence. |
| `resolved_length` | Full emitted sequence length. |
| `representability_status` | `representable`, `not_representable_too_long`, `not_representable_missing_constant`, `not_representable_invalid_sequence`, or `not_representable_unresolved_anchor`. |
| `failure_reason` | Required when not representable. |

The emitted USR rows should preserve Construct's existing lineage fields,
including `construct__context_id`, `construct__context_kind`,
`construct__template_id`, `construct__anchor_id`,
`construct__anchor_orientation`, `construct__anchor_start`,
`construct__anchor_end`, `construct__resolved_length`,
`construct__parts`, `construct__assembly_mode`, `construct__slot_count`,
`construct__slots`, and `construct__spec_id`. `construct__anchor_start` /
`construct__anchor_end` describe the focal slot for existing `anchor_mean`
consumers. RT and lnRNA subspans remain separately auditable through
`construct__slots` and study-owned view semantics, not through one collapsed
dual-cassette anchor.

Use a standardized regional context of 1,600 bp only if the full lnRNA and RT
cassettes fit without truncation. Do not truncate biologically meaningful RT or
lnRNA sequence to force comparability. If a candidate does not fit the declared
view, mark it as not representable under that view version and either exclude
it or require a new representation contract version.

Padding must use real plasmid prefix/suffix sequence. If there is not enough
real flanking sequence to reach the requested length, the construct view should
fail rather than pad with synthetic filler.

### 6. Representation And View Contract

The representation contract is:

```yaml
representation_contract: dual_cassette_construct_context_embedding_v1
```

Start with exactly three views:

```text
dual_cassette_1600bp_seq_mean
dual_cassette_1600bp_fwd_rc_concat
lnrna_span_in_construct_anchor_mean
```

These are representation views, not necessarily one USR sequence-view row each.
`dual_cassette_1600bp_seq_mean` maps to one forward `realized_context` sequence
view. `dual_cassette_1600bp_fwd_rc_concat` maps to two `realized_context`
sequence views plus one explicit derived vector view. The third is diagnostic
and must still be computed from the lnRNA span inside the full construct
context, not as an independent msd-only selector space.

All three views should be represented with existing USR sequence-view
vocabulary. The full construct context rows are `product_kind:
realized_context`, `context_kind: template_custom`, with `orientation:
forward` or `reverse_complement`. Study-specific semantics such as
`working_anchor`, `failed_anchor`, `khan_reference`, or
`crawford_reference` belong in `_views/view_semantics.parquet` or candidate
metadata, not in `product_kind`.

`dual_cassette_1600bp_seq_mean`, `dual_cassette_1600bp_fwd_rc_concat`, and
`lnrna_span_in_construct_anchor_mean` are study representation-view names.
They are not new USR `product_kind` values. When persisted against current USR
contracts, use `SequenceViewRecord.view_name` or `aliases` for the source view
label and use `_views/view_semantics.parquet` for study collection membership,
anchor roles, source regimes, and other mutable interpretation. Spans that
describe RT CDS, lnRNA, promoters, terminators, interstitial region, prefix,
and suffix are part of the construct projection contract or a study-owned
view-semantics/fixture table; do not add them as ad hoc columns to
`_views/sequence_views.parquet` unless USR first exposes a public extension
field.

Do not start with RT-only, msd-only, or broad atlas views as decision surfaces.
They may be added later as diagnostics after v1 proves useful, but the
experimental candidate row is the paired complete construct.

#### View: `dual_cassette_1600bp_seq_mean`

| Attribute | Contract |
| --- | --- |
| Alphabet | `dna_4`, uppercase A/C/G/T before Evo2 tokenization. |
| Coordinate basis | 0-based half-open coordinates over emitted forward construct sequence. |
| Sequence window | Full standardized 1,600 bp construct regional context. |
| Length policy | Exactly 1,600 bp only when all constants plus lnRNA plus RT CDS fit without truncation. |
| Strand/orientation | Forward emitted construct sequence, with lnRNA cassette before RT cassette. |
| Pooling | `seq_mean` over the full emitted sequence. |
| Variable spans | `lnRNA/msr-msd-payload` and `RT_CDS` recorded as explicit spans in construct order. |
| Constant spans | Plasmid prefix/suffix, promoters, RBS, terminators, interstitial region recorded as explicit spans. |
| Transform version | `dual_cassette_construct_context_embedding_v1`. |
| Failure behavior | Missing constants, invalid alphabet, length overflow, anchor clipping, or unresolved anchors produce non-representable rows before Infer. |

#### View: `dual_cassette_1600bp_fwd_rc_concat`

| Attribute | Contract |
| --- | --- |
| Alphabet | `dna_4`, uppercase A/C/G/T before Evo2 tokenization. |
| Coordinate basis | 0-based half-open coordinates over each emitted orientation-specific sequence. |
| Sequence window | Full standardized 1,600 bp construct regional context. |
| Length policy | Same as `dual_cassette_1600bp_seq_mean`. |
| Strand/orientation | Two declared views: forward and reverse-complement. The forward row uses lnRNA-before-RT cassette order; the reverse-complement row must already contain the reverse-complement sequence and orientation-specific pooling bounds. |
| Pooling | `seq_mean` over each full sequence, then explicit equal-block concatenation in a downstream feature/view layer. |
| Variable spans | `lnRNA/msr-msd-payload` and `RT_CDS` spans recorded in both emitted orientations. |
| Constant spans | Same as forward view, orientation-aware. |
| Transform version | `dual_cassette_construct_context_embedding_v1`. |
| Failure behavior | Either orientation missing or stale fails the concat alias. Infer must not reverse-complement a forward row implicitly. |

For repo congruency, this concat should be expressed the same way the current
LatentDNA/OPAL handoff treats bidirectional causal Evo2 summaries: separate
forward and reverse-complement Infer aliases first, then an explicit derived
view such as `block_normalized_concatenate` or an equivalently documented
export transform. The resulting vector is a row-level two-orientation summary,
not a native bidirectional Evo2 hidden state and not a raw, unlabeled Infer
vector splice.

#### View: `lnrna_span_in_construct_anchor_mean`

| Attribute | Contract |
| --- | --- |
| Alphabet | `dna_4`, uppercase A/C/G/T before Evo2 tokenization. |
| Coordinate basis | 0-based half-open coordinates over the full emitted construct sequence. |
| Sequence window | Full standardized 1,600 bp construct regional context. |
| Length policy | Same as `dual_cassette_1600bp_seq_mean`. |
| Strand/orientation | Forward emitted construct sequence for v1. Reverse-complement diagnostic may be v2. |
| Pooling | `anchor_mean` over the lnRNA span after running Evo2 on the full construct sequence. |
| Variable spans | lnRNA span is the pooling span; RT span remains present as paired context. |
| Constant spans | Same fixed construct constants. |
| Transform version | `dual_cassette_construct_context_embedding_v1`. |
| Failure behavior | Missing lnRNA span or clipped span fails before Infer. This view must not be generated from a naked lnRNA-only input. |

The diagnostic lnRNA-span view can be represented as a distinct sequence-view
row over the same emitted construct sequence by setting `anchor_start_0` /
`anchor_end_0` to the lnRNA span and `recommended_pooling: anchor_mean`. That
keeps Infer's existing single-span pooling contract intact. If later work needs
RT-span diagnostics too, add a separate diagnostic view row with RT pooling
bounds rather than overloading one view with multiple anchor spans.

Model outputs attach to declared `ConstructContextView` / `InferFeatureAlias`
sidecars. They do not attach directly to source records, raw candidates, or
literature rows.

### 7. Source And Reference Overlay Contracts

Use three distinct label/result layers:

```text
AbundancePriorOverlay      # literature/source msDNA or RT-DNA abundance
SpongingAssayObservation   # actual lab TF-sponging experimental Y
InferFeatureAlias          # model-derived X representation
```

This distinction is mandatory. Abundance priors are not sponging labels, and
embeddings are not assay observations.

#### Khan Overlay

```yaml
reference_overlay_id: khan_cross_retron_rt_dna_abundance_v1
regime: cross_retron_census
label_kind: rt_dna_abundance_relative_to_eco1
sequence_scope: rt_plus_ncrna_system
primary_use: high-producing non-Eco1 reference overlay
```

Current handoff paths from the sibling `dnadesign-data` tree:

| Artifact | Path | Data rows | Role |
| --- | --- | ---: | --- |
| Abundance overlay | `dnadesign-data/sources/literature/Khan_et_al_2024_retron_census/processed/overlays/abundance_prior_overlay.tsv` | 99 | Sparse cross-retron RT-DNA abundance prior. |
| Reference rows | `dnadesign-data/sources/literature/Khan_et_al_2024_retron_census/processed/references/rt_lnrna_references.tsv` | 171 | RT-lnRNA reference/provenance bridge. |

Verified current columns on 2026-05-22:

- abundance overlay includes `abundance_prior_id`, `reference_overlay_id`,
  `regime`, `label_kind`, `maps_to_reference_record_id`, `raw_value`,
  `normalized_value`, `ordinal_bin`, `censoring_status`, `construct_context`,
  `sequence_scope`, and `maps_to_view_id`.
- reference rows include RT provenance fields, lnRNA sequence/provenance fields,
  msr/msd/product sequence fields, `payload_program_id`,
  `variant_derivation`, and `construct_projection_status`.

Label fields:

- `raw_value`
- `normalized_value`
- `ordinal_bin`
- `censoring_status`

Semantics:

- sparse cross-retron RT + ncRNA census;
- `raw_value` / `normalized_value` are RT-DNA production relative to Eco1;
- non-detect or left-censored values must remain explicit;
- high-producing non-Eco1 systems are reference overlays, not default
  candidate generators and not TF-sponging labels.

#### Crawford Overlay

```yaml
reference_overlay_id: crawford_eco1_lnrna_msd_abundance_v1
regime: eco1_local_variant_library
label_kind: msdna_abundance
primary_use: dense Eco1-local abundance-prior overlay
```

Current handoff paths from the sibling `dnadesign-data` tree:

| Artifact | Path | Data rows | Role |
| --- | --- | ---: | --- |
| Abundance overlay | `dnadesign-data/sources/literature/Crawford_et_al_2025_retron_ncRNA_ML/processed/overlays/eco1_ncrna_abundance_observations.tsv` | 4,174 | Numeric Eco1 ncRNA abundance observations. |
| Handoff table | `dnadesign-data/sources/literature/Crawford_et_al_2025_retron_ncRNA_ML/processed/handoff/eco1_ncrna_abundance_handoff.tsv` | 4,174 | dnadesign-facing abundance handoff. |
| Design reference | `dnadesign-data/sources/literature/Crawford_et_al_2025_retron_ncRNA_ML/processed/references/eco1_lnrna_msd_designs.tsv` | 2,578 | Eco1-local lnRNA/MSD sequence/design reference rows. |

Verified current columns on 2026-05-22:

- abundance observations include `observation_id`, `reference_overlay_id`,
  `regime`, `label_kind`, `source_part_name`, `lnrna_sequence`,
  `raw_value`, `normalized_value`, `derived_log2_score`,
  `derived_rank_int_score`, `canonical_lnrna_sequence_id`,
  `duplicate_group_size`, `duplicate_status`, and
  `construct_projection_status`.
- handoff rows include `dnadesign_handoff_id`, `semantic_unit`,
  `part_name_for_dnadesign`, `numeric_score`, `numeric_score_semantics`,
  `canonical_lnrna_sequence_id`, `join_method`, `matched_design_record_id`,
  and `construct_projection_status`.
- design reference rows use the same RT/lnRNA/MSR/MSD provenance-style column
  family as the Khan reference bridge, scoped to Eco1-local designs.

Label fields:

- `raw_value`
- `normalized_value`
- `numeric_score` if materialized downstream from the handoff naming layer
- `derived_log2_score`
- `derived_rank_int_score`

Semantics:

- dense Eco1-local lnRNA/ncRNA/MSD variant abundance landscape;
- source score is enrichment/depletion relative to mean WT, with WT = 100;
- this is the stronger local abundance-prior overlay for retron26/retron43
  adjacent reasoning;
- it is still abundance, not TetR sponging.

Do not pool Khan and Crawford into one universal label. Use Crawford to ask
whether Eco1-local abundance has construct-context geometry. Use Khan to ask
whether high-producing cross-retron reference systems remain interpretable
after projection into the same synthetic construct grammar.

### 8. Candidate Table Schema

The candidate table should be one row per
`SyntheticRtLnrnaSpongingConstruct`. A first table can be TSV or Parquet, but
the typed contract should be table-shaped and convertible to a USR-backed
records surface.

When exported as an OPAL-ready USR-shaped `records.parquet`, `candidate_id`
must be copied to the required `id` column. The table must also expose the
canonical USR/OPAL base columns `bio_type`, `sequence`, `alphabet`, and
`length` for the emitted construct sequence. Study metadata should remain in
namespaced columns or companion sidecars so OPAL can validate the candidate
table without knowing this study's ontology.

Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `candidate_id` | string | Stable study candidate id. |
| `study_id` | string | `rt_lnrna_sponging_construct_triage`. |
| `construct_contract` | string | `dual_cassette_rt_lnrna_expression_v1`. |
| `representation_contract` | string | `dual_cassette_construct_context_embedding_v1`. |
| `payload_program_id` | string | `tetO_sponging_v1` for v1 unless a future payload program is explicitly declared. |
| `rt_cds_sequence_id` | string | USR sequence id or source-backed sequence id for RT CDS. |
| `rt_source` | string | Eco1, Khan source retron, catalytic-dead Eco1 derivative, etc. |
| `rt_variant` | string | `WT`, mutation token, catalytic-dead token, or source variant. |
| `rt_protein_provenance_id` | string | Provenance row id for encoded RT protein. |
| `lnrna_sequence_id` | string | USR sequence id or source-backed lnRNA sequence id. |
| `lnrna_source` | string | `Eco1-derived`, Crawford row source, Khan source retron, etc. |
| `lnrna_design_id` | string | `retron26`, `retron43`, compiler design id, Crawford design id, or source row id. |
| `lnrna_provenance_id` | string | Provenance row id for lnRNA sequence/design. |
| `msd_design_spec_id` | string/null | Study/compiler design spec id when the candidate has an engineered MSD/hairpin/payload spec. |
| `variant_derivation` | string/null | Optional explanation of how the row was generated. Not a required ontology class. |
| `source_basis` | string | Controlled value: `lab_anchor`, `compiler_variant`, `crawford_reference`, `khan_reference`, `rt_control`, or `manual_control`. |
| `construct_projection_status` | string | `pending`, `representable`, `not_representable`, or `excluded`. |
| `construct_id` | string/null | Filled after Construct projection. |
| `construct_context_view_ids` | list/string | Declared view ids once produced. |
| `abundance_prior_overlay_ids` | list/string | Khan/Crawford overlay ids attached by declared rules. |
| `sponging_assay_observation_ids` | list/string/null | Empty before lab labels. |
| `candidate_note` | string/null | Short human rationale. |

`source_basis` should be controlled, but it is not the removed
`perturbation_class` concept under another name. It records the immediate row
source so joins and rationale remain auditable:

| `source_basis` value | Meaning |
| --- | --- |
| `lab_anchor` | Current lab Eco1 WT RT plus a named Eco1-derived lnRNA design such as retron26 or retron43. |
| `compiler_variant` | Candidate generated from checked-in Snapback/scar-nick/retron MSD compiler primitives. |
| `crawford_reference` | Eco1-local lnRNA/MSD row linked to Crawford abundance/design references. |
| `khan_reference` | Cross-retron RT + ncRNA reference linked to Khan abundance/provenance rows. |
| `rt_control` | RT sequence control such as catalytic-dead Eco1, with lnRNA held by an explicit candidate rule. |
| `manual_control` | A deliberately included confound/control row with documented rationale. |

`variant_derivation` can remain null when the candidate is a source-backed
anchor whose structured fields already explain the row. It must be populated
when a row is generated by a design transform, mutation, or control rule.

Recommended metadata columns:

- `anchor_role`: `working_anchor`, `failed_anchor`, `weak_rescue_anchor`,
  `negative_control`, `reference_overlay`, `candidate`, or null.
- `rt_catalytic_status`: `wild_type`, `catalytic_dead`, `engineered`, or
  `source_reference`.
- `lnrna_region_focus`: `msd_p4`, `payload`, `cap`, `scar_nick`, `whole_lnrna`,
  or null.
- `length_nt_rt_cds`
- `length_nt_lnrna`
- `length_nt_construct`
- `gc_fraction_construct`
- `gc_fraction_lnrna`
- `repeat_burden`
- `inverted_repeat_burden`
- `stem_length_estimate`
- `codon_policy`

The inclusion rule is:

```text
include only if it helps test or diversify programmable multicopy ssDNA production in the synthetic dual-cassette context
```

### 9. MsdDesignSpec Schema And Compiler Mapping

`MsdDesignSpec` owns MSD/hairpin-level design and cloning feasibility
constraints. It is not a separate top-level `cloning_constraint_set`.

Required fields:

| Field | Meaning |
| --- | --- |
| `msd_design_spec_id` | Stable id for the design spec. |
| `schema_version` | `msd_design_spec_v1`. |
| `payload_program_id` | `tetO_sponging_v1` for v1. |
| `payload_id` | Concrete payload identifier, for example `tetO` or future bifunctional payload id. |
| `payload_sequence` | Payload sequence used in the MSD design. |
| `compiler_source` | Existing compiler contract or source path. |
| `compiler_contract` | For current designs, `retron_msd_compiler_spec_v1` or successor. |
| `cap_id` | Snapback cap/source id such as `C26`, `C43`, `C172`, etc. |
| `snapback_source_id` | Source primitive or cap-source lookup id. |
| `scar_nick_source_id` | Scar-nick primitive/source route id when applicable. |
| `left_base` | Left base/junction token when applicable. |
| `right_base` | Right base/junction token when applicable. |
| `profile_s3s2s1s0` | Existing scar-nick profile token where applicable. |
| `nick_orientation` | Optional route metadata. |
| `nickase` | Optional route metadata. |
| `feasibility_status` | `cloning_feasible`, `non_ligatable_control`, `needs_review`, or `not_feasible`. |
| `feasibility_reason` | Required for non-feasible or special controls. |
| `source_refs` | Paths or ids for source registry/design-set/provenance rows. |

Existing compiler primitives should map into this object rather than into a
new construct-level constraint ontology:

- `docs/studies/retron_hairpin_design/compiler/inputs/msd_design_177_194_cap_sources_spec.yaml`
  supplies typed labels under `retron_msd_compiler_spec_v1`.
- `docs/studies/retron_hairpin_design/workbench/design_sets/scar_nick_profile_panel_v1.yaml`
  supplies study-owned design-set rationale and resolved/non-ligatable control
  metadata.
- `docs/studies/retron_hairpin_design/compiler/catalog/msd_design_registry.yaml`
  supplies payload, cap, Snapback, and scar-nick route metadata.

The v1 workbench should primarily use Eco1-derived lnRNA/MSD variants. Broad
RT protein engineering and RT DMS must not dominate the v1 search space.

If existing retron hairpin compiler records use legacy lab-facing payload
tokens such as `TetR` for a design whose payload sequence is TetO, Phase 0
should normalize the distinction explicitly: `payload_program_id:
tetO_sponging_v1` is the program, `payload_id` is the concrete payload
sequence label, and any legacy token remains provenance or alias metadata. Do
not let payload naming drift become a new ontology branch.

### 10. Infer Handoff And Feature Sidecar Schema

Infer consumes sequence views. The study should create an explicit Infer
config for the three v1 views and then rely on Infer's feature alias sidecars.

The handoff should produce:

| Artifact | Owner | Purpose |
| --- | --- | --- |
| `sequence_views.parquet` rows | USR/Construct | Declared construct-context views with spans and orientations. |
| Infer extract config | Study/Infer | Declares Evo2 model, feature bundle, view selectors, pooling, and outputs. |
| `_derived/infer/feature_aliases.parquet` | Infer | View-aware vector aliases. |
| `_derived/infer/feature_vectors.parquet` | Infer | Reusable feature vector payloads. |
| `_derived/infer/feature_scalar_aliases.parquet` | Infer | Optional scalar aliases if diagnostics are enabled. |
| `_derived/infer/feature_scalars.parquet` | Infer | Optional scalar payloads. |
| LatentDNA workspace config | Study/LatentDNA | Declares source-backed Infer sidecar views, derived forward/RC concat, landmarks, metrics, plots, notebooks, and exports. |
| LatentDNA materialized views/metrics | LatentDNA | Review surface for geometry, anchor distances, overlay enrichment, and confound reports. |
| OPAL-ready candidate table | Study/OPAL prep | One selected fixed-length vector-valued `X` column for future OPAL use. |

Current Infer vector alias sidecars are the concrete persistence surface for
`InferFeatureAlias`. They live at `_derived/infer/feature_aliases.parquet` and
join to `_derived/infer/feature_vectors.parquet` through
`feature_vector_key`. The sidecar rows already carry the model/view/pooling
contract and should not be wrapped in a parallel study-owned result table.

Current vector alias rows should carry:

| Field | Meaning |
| --- | --- |
| `alias_id` | Stable alias row id for this view/vector/representation binding. |
| `view_id` | Declared USR sequence-view id used as model input. |
| `view_name` | Human-readable view label when present. |
| `sequence_id` | Base USR sequence id for the view. |
| `feature_vector_key` | Stable key to vector payload in `feature_vectors.parquet`. |
| `forward_pass_key` | Deduplication key for a reusable Evo2 forward pass. |
| `provider` | Infer provider, for example `evo2`. |
| `model_name` | Example `evo2_7b`. |
| `model_revision` | Provider/model revision if known. |
| `layer_name` | Canonical selector such as `block26_mlp_out`, not provider-private layer prose. |
| `representation_kind` | Feature family such as intermediate embedding or output-layer mean. |
| `pooling_operation` | `seq_mean` or `anchor_mean` for v1 construct views. Concat is a LatentDNA/export derived view, not an Infer pooling operation. |
| `pooling_start_0`, `pooling_end_0` | Emitted-orientation pooling bounds when applicable. |
| `orientation` | `forward` or `reverse_complement` from the source sequence view. |
| `source_dataset_id` | Owning USR dataset id. |
| `feature_request_digest` | Digest from Infer request/config. |
| `runtime_fingerprint_key` | Runtime contract key for stale/reusable detection. |
| `sequence_case_policy` | Expected to match the current uppercase DNA case policy. |
| `created_at` | Alias creation timestamp. |

Candidate ids and `construct_context_view_id` are recovered by joining the
alias row back through the view/candidate metadata. OPAL-facing column names
are chosen only when a study or export materializes a candidate table.

The OPAL/export mapping should carry:

| Field | Meaning |
| --- | --- |
| `candidate_id` | Candidate row id after joining alias/view metadata to the candidate table. |
| `construct_context_view_id` | The USR `view_id` or study alias for the selected construct context view. |
| `x_column_name` | OPAL-ready column name in `records.parquet`, for example `rt_lnrna_construct_embedding`. |
| `x_dim` | Fixed vector length after any LatentDNA/export transform. |
| `x_source_view_id` | Source Infer or LatentDNA view id used to build the column. |
| `x_materialization_status` | `planned`, `materialized`, `stale`, or `failed`. |

Likelihood outputs may be collected as diagnostics only if explicitly declared.
They must not become the primary ranking objective because variable-length
inverted-repeat/hairpin sequences can make autoregressive likelihood misleading.

LatentDNA should be the default home for representation review once Infer
sidecars exist. Its source should be `kind: infer_feature_sidecar` for vectors
and `kind: infer_feature_scalar_sidecar` only for explicit scalar diagnostics.
Views should use `vector.kind: column` with `name: value` against those
sidecar sources. The forward/RC representation should use an explicit derived
view, preferably `block_normalized_concatenate`, so the OPAL-facing `X` is a
documented fixed-length vector rather than an implicit concat performed inside
OPAL or a notebook.

### 11. Abundance Priors Versus Sponging Labels

Do not conflate the three result types:

| Layer | Owned by | Meaning | May drive OPAL? |
| --- | --- | --- | --- |
| `AbundancePriorOverlay` | Source handoff / USR overlay | Literature/source msDNA or RT-DNA abundance. | No, not as TF-sponging `Y`; it can be metadata or a separate analysis target. |
| `InferFeatureAlias` | Infer | Model-derived `X` representation for a declared construct view. | Yes, as `X` after materialization. |
| `SpongingAssayObservation` | Study / lab / OPAL label sidecar | Actual TF-sponging experimental label and assay metadata. | Yes, as `Y` after labels exist. |

`SpongingAssayObservation` schema:

| Field | Meaning |
| --- | --- |
| `observation_id` | Stable label row id. |
| `candidate_id` | Candidate construct id. |
| `payload_program_id` | Payload program assayed, for example `tetO_sponging_v1`. |
| `assay_id` | Assay protocol/run id. |
| `batch_id` | Experimental batch id. |
| `readout_kind` | `TF_sponging_reporter`, `growth_normalized_reporter`, `mature_ssdna_abundance`, etc. |
| `raw_value` | Raw measured value. |
| `normalized_value` | Normalized scalar used as candidate `Y`, when defined. |
| `normalization_basis` | Controls/reference used for normalization. |
| `replicate_count` | Number of replicates or source count. |
| `uncertainty` | Standard error, SD, confidence interval, or null. |
| `assay_metadata` | Host, induction, reporter, plate/run metadata, and caveats. |

Before real sponging labels exist, OPAL may help organize candidates but should
not act as a learned selector.

### 12. OPAL Readiness And Label Lifecycle

The candidate table should be OPAL-ready before labels exist, but an OPAL
campaign should start only after real labels exist. Future OPAL rows should
look like:

```text
candidate_id
construct_context_view_id
x_column = rt_lnrna_construct_embedding
variant_derivation
design_constraints
abundance_prior_overlay_ids
y_column = normalized_TF_sponging_label optional
assay_metadata
batch_id
```

`design_constraints` in OPAL export is a flattened metadata summary from
`MsdDesignSpec`; it is not a separate core ontology object.

The OPAL handoff contract is:

- one candidate universe with stable ids in `records.parquet`;
- one explicit fixed-length infer/LatentDNA-derived vector-valued `X` column
  selected by the study and stored as an Arrow fixed-size-list float vector;
- pre-campaign export may have an absent or empty label slot;
- future shared label sidecar for `SpongingAssayObservation` rows, preferably
  dataset-local `_opal/observed_labels.parquet` with `labels.source.kind:
  usr_sidecar`;
- campaign-scoped predictions and selections, not primary label duplication;
- `writeback.prediction_records: ledger_only` for shared-label campaigns;
- no learned `Y ~ X` loop until real sponging labels are ingested.

Before the first real assay ingest, a study-level readiness check may describe
an empty future label slot, but an actual OPAL `run`/`explain` campaign should
not start unless the configured label source satisfies OPAL's own contract.
For `usr_sidecar`, that means the sidecar path is explicit, campaign configs
declare `writeback.prediction_records`, and any existing rows contain only ids
present in the candidate universe. A missing or empty sidecar is allowed only
for validation/pre-ingest planning; it is not evidence that a supervised round
can train.

After labels arrive, OPAL can learn `Y ~ X` for complete paired RT-lnRNA
constructs. It must not be trained on RT-only or msd-only representations as if
they were the same experimental object.

The OPAL candidate table should preserve enough metadata for audit and review,
but OPAL should not be the place where construct representability, abundance
overlay joins, or representation selection are decided. Those decisions belong
upstream in the study, Construct, Infer, and LatentDNA surfaces. OPAL validation
should only see the final campaign contract: candidate ids, required USR base
columns, one finite fixed-size vector `X`, and a label source that is empty
before first assay ingest.

### 13. Analysis Metrics And Pre-OPAL Triage Gates

Before OPAL has labels, ranking is constrained triage, not prediction. Do not
define a universal score yet.

Required gates before using embeddings for candidate selection:

1. Candidate satisfies design/cloning constraints through `MsdDesignSpec` or
   explicit source compatibility.
2. Candidate fits the declared construct view without truncation.
3. Candidate has explicit RT and lnRNA provenance.
4. Candidate has a clear `variant_derivation` or source basis.
5. Candidate passes basic confound checks or is explicitly used as a
   confound/control.

Core geometry metrics:

```text
d_to_retron26_working
d_to_retron43_failed
working_failed_axis_coordinate
off_axis_distance
```

For an embedding vector `x`, working anchor `w`, and failed anchor `f`:

```text
d_to_retron26_working = cosine_distance(x, w)
d_to_retron43_failed = cosine_distance(x, f)
axis = normalize(f - w)
working_failed_axis_coordinate = dot(x - w, axis)
off_axis_distance = norm((x - w) - working_failed_axis_coordinate * axis)
```

Use the same vector space and same declared view for all terms in one metric.
Do not compare `dual_cassette_1600bp_seq_mean` distances to
`lnrna_span_in_construct_anchor_mean` distances as if they were the same axis.

Abundance overlay metrics:

```text
abundance_bin_neighbor_enrichment
abundance_axis_coordinate
spearman_embedding_axis_vs_ordinal_abundance
```

Rules:

- Compute overlay-specific abundance metrics only within a declared regime.
- For Khan, use ordinal bins and account for non-detect/left-censored values.
- For Crawford, use continuous values plus ordinal bins.
- Do not average Khan and Crawford labels into a single abundance target.
- Do not overemphasize "orthogonal novelty," "class diversity," or mandatory
  perturbation classes. Candidate rationale can use optional
  `variant_derivation`, but selection should remain hypothesis-driven and
  readable.

Minimum confound checks:

- construct length;
- lnRNA length;
- GC fraction;
- repeat burden;
- inverted-repeat burden;
- stem length estimate when available;
- codon policy for RT CDS rows;
- source/regime effects for overlays;
- payload program id.

### 14. Success And Failure Criteria

The representation contract is useful for triage only if:

- retron26 and retron43 do not collapse in embedding space;
- weak/rescue anchors land plausibly relative to the working/failed axis;
- Crawford abundance bins show non-random local structure;
- Khan high producers are interpretable as external overlays beyond simple
  clade/source effects;
- length, GC, repeat burden, stem length, inverted-repeat burden, and codon
  policy do not explain the whole geometry;
- cloning-feasible compiler variants stratify non-randomly;
- selected candidates can be described as testing specific biological
  hypotheses.

If retron26 and retron43 collapse, or if confounds explain the geometry, the
embedding surface is descriptive only and must not drive wet-lab triage. In
that case, candidate selection should fall back to mechanistic design logic and
explicit controls.

### 15. Non-goals And Anti-patterns

This spec explicitly excludes:

- a generic atlas/browser of all retrons;
- full enumeration of all predicted retron-associated RTs;
- making Infer fetch databases;
- making Permuter own source discovery;
- putting raw crawling state inside the existing retron hairpin study;
- treating Evo2 likelihood as a primary ranking objective;
- treating RT-only or msd-only embeddings as the main experimental selector;
- treating Khan/Crawford abundance as TF-sponging labels;
- treating source records as construct views;
- treating model outputs as assay observations;
- over-annotating branch-G/msr/msd/stem/loop boundaries as mandatory v1 schema
  fields;
- making Eco1 the ontology rather than the v1 anchor;
- broad payload search without a declared payload program;
- broad RT protein engineering as the main v1 objective;
- cross-pairing unrelated RTs and lnRNAs by default;
- silently truncating RT or lnRNA sequence to fit a fixed view;
- silently falling back to shorter/alternate contexts when a view cannot be
  represented.

### 16. Implementation Phases

#### Phase 0: Audit And Contracts

Define schemas, ids, naming, and file paths. Confirm:

- exact dual-cassette plasmid constants;
- construct template source sequence and real prefix/suffix padding;
- public multi-slot Construct contract for lnRNA and RT CDS placement;
- retron26/retron43 sequence sources and naming normalization;
- current Khan and Crawford handoff files and row counts;
- whether any weak/rescue anchors are available;
- expected first study record location.

Deliverables:

- candidate table schema fixture;
- `MsdDesignSpec` schema fixture;
- construct contract fixture;
- representation view contract fixture;
- overlay mapping contract fixture;
- study record bootstrap plan.

#### Phase 1: Candidate Table

Create `SyntheticRtLnrnaSpongingConstruct` rows for:

- Eco1 WT RT + retron26 lnRNA working anchor;
- Eco1 WT RT + retron43 lnRNA failed P4/stem-extension anchor;
- available weak/rescue anchors;
- finite cloning-feasible MSD variants from existing Snapback/scar-nick/compiler
  primitives;
- catalytic-dead RT negative control if selected for v1;
- construct-compatible high-producing non-Eco1 references if selected for v1.

Deliverables:

- checked-in or generated candidate table with stable ids;
- provenance table for RT and lnRNA records;
- `MsdDesignSpec` records for engineered MSD variants;
- validation report for missing provenance or duplicate ids.

#### Phase 2: Construct Projection

Use Construct to produce declared dual-cassette construct views. Record
non-representable candidates rather than truncating.

Deliverables:

- Construct workspace/config or neutral construct projection manifest;
- documented projection strategy: public `construct_multi_slot_assembly_v1`
  with named lnRNA and RT CDS slots;
- USR records for realized construct sequences;
- `_views/sequence_views.parquet` rows for the three representation views,
  using `realized_context` product kind for emitted construct contexts;
- `_views/view_semantics.parquet` rows for study role tags, source family,
  and candidate/view collections;
- representability report.

#### Phase 3: Infer Handoff

Generate Infer configs for the three v1 embedding views and write
`InferFeatureAlias`/sidecar references.

Deliverables:

- Infer sequence-view completion validation;
- feature sidecars for all representable candidate/view pairs;
- explicit OPAL-ready `X` column selection;
- stale/missing feature report.

#### Phase 4: Overlay Linkage

Attach Khan and Crawford abundance-prior overlays by declared mapping rules.
Keep overlay labels separate from sponging labels.

Deliverables:

- overlay join report;
- unmatched source rows report;
- left-censor handling report for Khan;
- Crawford ordinal-bin derivation report.

#### Phase 5: Analysis

Compute distances to working/failed anchors, abundance overlay enrichment/axis
metrics, and confound checks.

Deliverables:

- LatentDNA workspace config or equivalent study-owned analysis config;
- materialized LatentDNA source-backed views from Infer sidecars;
- derived forward/RC concat view if selected for the primary `X`;
- metrics table;
- confound audit table;
- short analysis report stating whether representation geometry is usable for
  constrained triage.

#### Phase 6: Wet-lab Handoff

Generate a small candidate list by rationale category, not a single global
score.

Candidate rationale categories may include:

- close to retron26 productive anchor;
- between retron26 and retron43;
- near retron43 as negative/control rows;
- cloning-feasible variants spanning the working/failed axis;
- catalytic-dead RT control;
- high-producing non-Eco1 reference overlay if construct-compatible.

Deliverables:

- wet-lab handoff table;
- candidate rationale notes;
- required controls and readouts.

#### Phase 7: OPAL Readiness

Once real `SpongingAssayObservation` labels arrive, export OPAL-ready `X/Y`
matrices and metadata.

Deliverables:

- OPAL-ready candidate table with one explicit fixed-length `X` column;
- shared observed-label sidecar or equivalent label source;
- OPAL campaign config after labels exist;
- first label-ingest validation.

### 17. Validation Tests

Required validation should scale with the implementation phase:

| Area | Test |
| --- | --- |
| Docs | `uv run python -m dnadesign.devtools.docs.checks --repo-root .` for checked-in docs. |
| Boundaries | `uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .` when adding code/imports. |
| Candidate schema | Reject missing `candidate_id`, missing RT/lnRNA provenance, duplicate ids, and unknown `payload_program_id`. |
| Forbidden core fields | Candidate core schema must not require `perturbation_class`, `cloning_constraint_set`, or `representation_result`. |
| `MsdDesignSpec` | Validate compiler source, cap/base/profile tokens, feasibility status, and payload id. |
| Construct projection | Reject invalid alphabet, missing constants, anchor/slot clipping, arbitrary filler padding, and >1,600 bp views that would require truncation. Validate that implementation uses the checked public multi-slot Construct contract. |
| Sequence views | Validate product kind, orientation, construct spans, length, pooling bounds, transform version, and that no study-specific `product_kind` values were invented. |
| Infer completion | Run Infer sequence-view completion validation before model execution. |
| Feature aliases | Verify every materialized `InferFeatureAlias` points to one declared `ConstructContextView` and one payload sidecar. |
| LatentDNA review | Validate workspace config, materialize sidecar-backed views, and fail if derived concat views are stale, row-misaligned, or built from missing Infer aliases. |
| Overlay linkage | Verify Khan 99-row and Crawford 4,174-row abundance overlays retain raw numeric labels, normalized values, provenance, and regime ids. |
| Label separation | Verify no `AbundancePriorOverlay` column is exported as `normalized_TF_sponging_label`. |
| OPAL precondition | OPAL config validation requires one explicit Arrow fixed-size-list vector `X` column. A future or empty label slot is allowed for study readiness prose, but OPAL `run`/`explain` must use a configured label source that satisfies OPAL's `campaign_history` or `usr_sidecar` contract. |
| Metrics | Unit-test anchor distance formulas and fail when anchors are missing or from mismatched views. |
| Confounds | Emit a report that includes length, GC, repeat burden, inverted-repeat burden, stem length if available, and codon policy. |

Acceptance for the first useful v1 workbench:

- all v1 candidate rows have explicit RT and lnRNA provenance;
- retron26 and retron43 have materialized construct views;
- no candidate was silently truncated;
- all materialized embeddings are tied to declared construct-context views;
- Khan and Crawford overlays remain separate regimes;
- no TF-sponging label is faked from abundance or embedding data.

### 18. Open Questions

Preserve these as unresolved unless Phase 0 confirms them from repo contents or
lab records:

- Exact plasmid/context sequence and whether the 1,600 bp window is fixed or
  configurable.
- Exact promoter/RBS/terminator/interstitial annotations for the dual-cassette
  construct.
- Whether later construct contracts need additional slots beyond lnRNA and RT
  CDS after the v1 two-slot path is materialized.
- Exact source of retron26/retron43 sequences and naming normalization.
- Which weak/rescue anchors are already available.
- Whether Crawford abundance labels are directly joinable to
  retron26/retron43-adjacent candidate designs after sequence/design
  normalization and construct projection.
- Which Khan high-producing non-Eco1 retrons are construct-compatible.
- How to define ordinal abundance bins for Crawford and Khan.
- What confound features are required: length, GC, repeat burden,
  inverted-repeat score, stem length, codon policy.
- Whether bifunctional payloads enter v1 or remain a declared future payload
  program.
- Whether catalytic-dead RT controls are represented as candidate rows in v1 or
  only as future controls.
- Whether mature multicopy ssDNA abundance will be measured directly before
  TF-sponging labels.
- Exact OPAL `y_space` for first lab labels: scalar normalized TF-sponging
  score, vector-valued assay summary, or a staged abundance-then-sponging label
  lifecycle.
- Exact owner and path for the OPAL-ready `records.parquet`: study-generated
  candidate dataset, LatentDNA export bundle materialized into USR, or a
  purpose-built study helper.
- Whether the diagnostic lnRNA-span view needs a reverse-complement companion
  in v1 or should wait for v2.
- Whether construct-compatible non-Eco1 references should be actual candidate
  rows, overlay-only rows, or both with explicit anchor roles.

### 19. Distilled Plan

Build a dual-cassette RT-lnRNA construct representation study, anchored by
Eco1 WT RT with retron26 and retron43 lnRNA designs, using Crawford Eco1
abundance priors and Khan cross-retron abundance priors as small reference
overlays. Test whether standardized construct-context Evo2 embeddings contain
an ordinal multicopy ssDNA / RT-DNA abundance structure before using them to
triage programmable TF-sponging designs.

The implementation should be contract-first, source-provenanced,
view-explicit, OPAL-ready, and narrow enough to avoid becoming a generic
retron atlas.

### Appendix A. Competency Questions And Semantic Trace

Use these questions to keep the ontology operational rather than decorative.
Every class, field, and artifact in v1 should help answer at least one of
these questions.

| Competency question | Required terms/artifacts | Validation implication |
| --- | --- | --- |
| What is the experimental unit? | `SyntheticRtLnrnaSpongingConstruct`, `RtCds`, `LnrnaSequence`, `DualExpressionConstruct` | Candidate rows must be paired RT + lnRNA constructs, not RT-only, lnRNA-only, or source-record rows. |
| Where did the RT and lnRNA come from? | `RtProteinProvenance`, `LnrnaProvenance`, source ids, source rows | Candidate validation rejects missing RT or lnRNA provenance. |
| Which part of the row is engineered versus imported? | `source_basis`, optional `variant_derivation`, `MsdDesignSpec` | Candidate rationale is recoverable without a mandatory perturbation-class ontology. |
| Is the row construct-compatible? | `DualExpressionConstruct`, `ConstructContextView`, `representability_status` | Construct projection records failure rather than truncating or changing view versions silently. |
| What sequence did Evo2 actually see? | `ConstructContextView`, sequence-view rows, orientation and span metadata | Infer inputs must be traceable to one declared construct view. |
| Where is the embedding or feature vector? | `InferFeatureAlias`, feature sidecars, `x_column_name` | Model outputs attach to view aliases, not to source records or assay labels. |
| Which source abundance priors apply? | `AbundancePriorOverlay`, regime id, overlay join report | Khan and Crawford remain separate overlay regimes with raw values preserved. |
| Which rows have true TF-sponging labels? | `SpongingAssayObservation` | OPAL `Y` is absent until lab labels exist; abundance priors cannot be promoted to `Y`. |
| Can the representation support triage? | geometry metrics, overlay metrics, confound audit | Triage is blocked if anchors collapse or confounds explain the geometry. |
| What should be built next? | phase deliverables, open questions, validation tests | The next implementation slice has explicit entry and exit criteria. |

The model is intentionally table-first. A future RDF/JSON-LD surface can be
added only after the tabular contract proves stable; v1 should not introduce a
formal ontology runtime just to express a small candidate table.

### Appendix B. Contract Fixture Sketches

These sketches are not final checked-in fixture values. They show the shape
Phase 0 should materialize with real sequence ids, real construct constants,
and current source row ids.

#### Candidate Row Sketch

```yaml
candidate_id: rt_lnrna_triage__eco1_wt__retron26__tetO_v1
study_id: rt_lnrna_sponging_construct_triage
construct_contract: dual_cassette_rt_lnrna_expression_v1
representation_contract: dual_cassette_construct_context_embedding_v1
payload_program_id: tetO_sponging_v1
rt_cds_sequence_id: usr:sequence:pending_eco1_wt_rt_cds
rt_source: Eco1
rt_variant: WT
rt_protein_provenance_id: rt_prov__eco1_wt_current_lab
lnrna_sequence_id: usr:sequence:pending_eco1_derived_retron26_lnrna
lnrna_source: Eco1-derived
lnrna_design_id: retron26
lnrna_provenance_id: lnrna_prov__eco1_derived_retron26
msd_design_spec_id: msd_spec__retron26__tetO_sponging_v1
variant_derivation: null
source_basis: lab_anchor
anchor_role: working_anchor
construct_projection_status: pending
construct_id: null
construct_context_view_ids: []
abundance_prior_overlay_ids: []
sponging_assay_observation_ids: []
candidate_note: Working Eco1 WT RT plus Eco1-derived retron26 lnRNA anchor.
```

```yaml
candidate_id: rt_lnrna_triage__eco1_wt__retron43__tetO_v1
study_id: rt_lnrna_sponging_construct_triage
construct_contract: dual_cassette_rt_lnrna_expression_v1
representation_contract: dual_cassette_construct_context_embedding_v1
payload_program_id: tetO_sponging_v1
rt_cds_sequence_id: usr:sequence:pending_eco1_wt_rt_cds
rt_source: Eco1
rt_variant: WT
rt_protein_provenance_id: rt_prov__eco1_wt_current_lab
lnrna_sequence_id: usr:sequence:pending_eco1_derived_retron43_lnrna
lnrna_source: Eco1-derived
lnrna_design_id: retron43
lnrna_provenance_id: lnrna_prov__eco1_derived_retron43
msd_design_spec_id: msd_spec__retron43_p4_stem_extension__tetO_sponging_v1
variant_derivation: p4_stem_extension
source_basis: lab_anchor
anchor_role: failed_anchor
construct_projection_status: pending
construct_id: null
construct_context_view_ids: []
abundance_prior_overlay_ids: []
sponging_assay_observation_ids: []
candidate_note: Failed Eco1 WT RT plus Eco1-derived retron43 P4/stem-extension anchor.
```

#### MsdDesignSpec Sketch

```yaml
msd_design_spec_id: msd_spec__compiler__C172__AGTG_CAAT__MXMM__tetO_v1
schema_version: msd_design_spec_v1
payload_program_id: tetO_sponging_v1
payload_id: tetO
payload_sequence: pending_phase0_sequence
compiler_source: docs/studies/retron_hairpin_design/compiler
compiler_contract: retron_msd_compiler_spec_v1
cap_id: C172
snapback_source_id: snapback_cap__C172
scar_nick_source_id: scar_nick_profile_panel_v1
left_base: AGTG
right_base: CAAT
profile_s3s2s1s0: MXMM
nick_orientation: pending_phase0
nickase: pending_phase0
feasibility_status: cloning_feasible
feasibility_reason: null
source_refs:
  - docs/studies/retron_hairpin_design/compiler/catalog/msd_design_registry.yaml
  - docs/studies/retron_hairpin_design/workbench/design_sets/scar_nick_profile_panel_v1.yaml
```

`payload_id` is the concrete payload sequence label. `payload_program_id` is
the declared payload-design program. Bifunctional sponging should enter as a
future payload program, for example `bifunctional_tf_sponging_v1`, not as a
rewrite of the RT-lnRNA ontology.

#### ConstructContextView Sketch

This is a study contract fixture sketch, not the exact row shape of
`_views/sequence_views.parquet`. Current `SequenceViewRecord` rows should use
only public USR fields such as `view_id`, `sequence_id`, `view_name`,
`aliases`, `product_kind`, `context_kind`, `orientation`, lineage ids,
`anchor_start_0`, `anchor_end_0`, `forward_anchor_start_0`,
`forward_anchor_end_0`, and `recommended_pooling`. The `view_kind`,
`variable_spans`, `constant_spans`, and status fields below belong in a
study-owned contract fixture or view-semantics/representability sidecar unless
USR adds a public extension.

```yaml
construct_context_view_id: view__rt_lnrna_triage__candidate_001__dual_cassette_1600bp_fwd
candidate_id: rt_lnrna_triage__candidate_001
construct_id: construct__rt_lnrna_triage__candidate_001
representation_contract: dual_cassette_construct_context_embedding_v1
view_kind: dual_cassette_1600bp_seq_mean
product_kind: realized_context
context_kind: template_custom
alphabet: dna_4
coordinate_basis: zero_based_half_open
sequence_window: full_standardized_construct
target_length_nt: 1600
length_policy: exact_1600bp_real_plasmid_padding_no_truncation
orientation: forward
recommended_pooling: seq_mean
variable_spans:
  lnrna: [pending_start, pending_end]
  RT_CDS: [pending_start, pending_end]
constant_spans:
  plasmid_prefix: [0, pending_end]
  lnrna_promoter: [pending_start, pending_end]
  lnrna_terminator: [pending_start, pending_end]
  interstitial_region: [pending_start, pending_end]
  rt_promoter: [pending_start, pending_end]
  rt_rbs: [pending_start, pending_end]
  rt_terminator: [pending_start, pending_end]
  plasmid_suffix: [pending_start, 1600]
status: planned
failure_reason: null
```

#### InferFeatureAlias Sketch

```yaml
alias_id: alias_pending
view_id: view__rt_lnrna_triage__candidate_001__dual_cassette_1600bp_fwd
view_name: dual_cassette_1600bp_seq_mean
sequence_id: usr:sequence:pending_construct_context
feature_vector_key: pending_infer_vector_key
forward_pass_key: pending_forward_pass_key
provider: evo2
model_name: evo2_7b
model_revision: pending_model_revision
layer_name: block26_mlp_out
representation_kind: intermediate_embedding
pooling_operation: seq_mean
pooling_start_0: 0
pooling_end_0: 1600
orientation: forward
source_dataset_id: pending_usr_dataset
feature_request_digest: pending_digest
runtime_fingerprint_key: pending_runtime_fingerprint
sequence_case_policy: upper_acgt
created_at: pending_timestamp
```

```yaml
opal_x_export_binding:
  candidate_id: rt_lnrna_triage__candidate_001
  construct_context_view_id: view__rt_lnrna_triage__candidate_001__dual_cassette_1600bp_fwd
  x_column_name: rt_lnrna_construct_embedding
  x_source_view_id: intermediate_embedding_7b_dual_cassette_1600bp_seq_mean
  x_dim: 4096
  x_materialization_status: planned
```

#### AbundancePriorOverlay Sketch

```yaml
abundance_prior_id: khan_cross_retron_rt_dna_abundance_v1__source_row_001
reference_overlay_id: khan_cross_retron_rt_dna_abundance_v1
regime: cross_retron_census
label_kind: rt_dna_abundance_relative_to_eco1
sequence_scope: rt_plus_ncrna_system
raw_value: pending_source_value
normalized_value: pending_source_value
ordinal_bin: pending_bin
censoring_status: observed_or_left_censored
maps_to_reference_record_id: pending_khan_reference_record
maps_to_candidate_id: null
```

```yaml
abundance_prior_id: crawford_eco1_lnrna_msd_abundance_v1__observation_001
reference_overlay_id: crawford_eco1_lnrna_msd_abundance_v1
regime: eco1_local_variant_library
label_kind: msdna_abundance
sequence_scope: eco1_lnrna_variant
raw_value: pending_source_value
normalized_value: pending_source_value
numeric_score: pending_handoff_score
value_units: enrichment_depletion_relative_to_mean_wt_wt_100
maps_to_design_record_id: pending_crawford_design_record
maps_to_candidate_id: null
```

### Appendix C. Source Mapping And Join Rules

The source mapping should be conservative and auditable. A source row becomes a
candidate row only after the study explicitly projects it into the synthetic
dual-cassette construct contract.

#### Khan Join Rules

- Start from
  `dnadesign-data/sources/literature/Khan_et_al_2024_retron_census/processed/overlays/abundance_prior_overlay.tsv`.
- Preserve all 99 abundance-prior rows and keep `raw_value` and
  `normalized_value` as primary numeric payloads.
- Join to
  `dnadesign-data/sources/literature/Khan_et_al_2024_retron_census/processed/references/rt_lnrna_references.tsv`
  through `maps_to_reference_record_id` -> `record_id`.
- Treat `ordinal_bin` and `censoring_status` as analysis metadata. They do not
  replace the raw numeric source values.
- A Khan reference may become a candidate only when both RT CDS and lnRNA
  sequence are sufficient for `dual_cassette_rt_lnrna_expression_v1`, and when
  the emitted construct fits the declared representation view without
  truncation.
- High-producing non-Eco1 systems should default to `source_basis:
  khan_reference` and `anchor_role: reference_overlay`, unless Phase 0 promotes
  a specific row into the main candidate set with a written rationale.

#### Crawford Join Rules

- Start from
  `dnadesign-data/sources/literature/Crawford_et_al_2025_retron_ncRNA_ML/processed/overlays/eco1_ncrna_abundance_observations.tsv`.
- Preserve all 4,174 abundance observations and keep `raw_value` /
  `normalized_value` primary.
- Use
  `dnadesign-data/sources/literature/Crawford_et_al_2025_retron_ncRNA_ML/processed/handoff/eco1_ncrna_abundance_handoff.tsv`
  as the dnadesign-facing handoff layer when `numeric_score` naming is more
  convenient for candidate-table joins.
- Use `matched_design_record_id` to join handoff rows to
  `dnadesign-data/sources/literature/Crawford_et_al_2025_retron_ncRNA_ML/processed/references/eco1_lnrna_msd_designs.tsv`
  when available.
- If `matched_design_record_id` is missing but `canonical_lnrna_sequence_id`
  is present, record a sequence-based join separately and mark the join method.
- Do not collapse duplicate lnRNA sequence groups unless a downstream analysis
  explicitly chooses a duplicate policy. Keep `duplicate_status` and
  `duplicate_group_size` available to the confound audit.
- Crawford rows are Eco1-local abundance priors. They are not TetR sponging
  labels and not construct views.

#### Cross-source Rules

- Never average Khan and Crawford values into one label.
- Never join source rows to embeddings directly. The route is source row ->
  reference/provenance -> candidate row -> construct view -> Infer feature.
- Every overlay join report should include matched rows, unmatched rows,
  duplicate groups, missing sequence fields, and rows excluded because of
  construct representability.
- Any row promoted from overlay/reference status to candidate status must gain
  explicit RT provenance, lnRNA provenance, `source_basis`, and candidate
  rationale.

### Appendix D. Artifact Layout For Phase 0 And Phase 1

Phase 0 should produce small contract fixtures before any broad embedding run.
A concrete layout can be:

```text
docs/studies/rt_lnrna_sponging_construct_triage/
  README.md
  contexts/
    construct-contract.md
    representation-contract.md
    source-overlays.md
    opal-handoff.md
  operations/
    contract/
      schemas/
        candidate-table.schema.yaml
        msd-design-spec.schema.yaml
        construct-context-view.schema.yaml
        abundance-prior-overlay.schema.yaml
        sponging-assay-observation.schema.yaml
      fixtures/
        retron26-working-anchor.yaml
        retron43-failed-anchor.yaml
        crawford-overlay-minimal.yaml
        khan-overlay-minimal.yaml
      checks/
        candidate-table-validation.yaml
        representation-readiness.yaml
        overlay-linkage.yaml
  record/
    campaign.yaml          # optional until a LatentDNA study binding or OPAL campaign needs it
    datasets.yaml
    status.md
  workbench/
    design_sets/
      v1_candidate_scope.md
    ontology/
      vocabulary.md
    provenance/
      source-handoff-ledger.md
```

Generated or larger runtime artifacts should not be checked in by default.
They should live under an explicit workspace or output root, for example:

```text
workspaces/studies/rt_lnrna_sponging_construct_triage/
  candidates/
    synthetic_rt_lnrna_sponging_constructs.tsv
    msd_design_specs.tsv
    rt_provenance.tsv
    lnrna_provenance.tsv
  construct/
    construct_projection_manifest.yaml
    representability_report.tsv
    _views/sequence_views.parquet
  infer/
    configs/
    feature_aliases.parquet
    feature_vectors.parquet
  overlays/
    khan_overlay_join_report.tsv
    crawford_overlay_join_report.tsv
  analysis/
    candidate_geometry_metrics.tsv
    overlay_regime_metrics.tsv
    confound_audit.tsv
    triage_decision_report.md
  handoff/
    wet_lab_candidate_handoff.tsv
    opal_ready_candidate_table.parquet
```

If the representation review uses LatentDNA, the checked-in workspace config
may live under the owning tool surface, for example:

```text
src/dnadesign/latentdna/workspaces/rt_lnrna_sponging_construct_triage/
  config.yaml
  plot_semantics/
  outputs/              # generated; not hand-edited
```

The candidate table exported for OPAL should be a USR-shaped
`records.parquet` with one explicit fixed-size vector `X` column. The broader
LatentDNA review workspace can contain projections, metrics, notebooks, and
appendix views, but OPAL should not consume a notebook or UMAP as the campaign
contract.

If the implementation discovers that a reusable schema belongs in Construct,
USR, Infer, or OPAL, move only that reusable contract to the owning package.
Keep study-specific anchor names, rationale, and wet-lab categories in the
study surface.

### Appendix E. Minimal Vertical Slice

The smallest useful implementation is not the full candidate universe. It is a
contract proof that the anchors, views, overlays, and labels are separated
correctly.

1. Create Phase 0 schema fixtures for candidate rows, `MsdDesignSpec`,
   construct views, and overlay mappings.
2. Create two candidate rows: retron26 working anchor and retron43 failed
   anchor, both with Eco1 WT RT and explicit lnRNA provenance.
3. Add one catalytic-dead RT candidate only if the exact RT CDS edit and
   codon policy are already known; otherwise leave it in open questions.
4. Add a tiny Crawford overlay sample and a tiny Khan overlay sample without
   promoting those source rows to construct candidates.
5. Resolve the dual-cassette plasmid constants and 1,600 bp padding source.
6. Use the public multi-slot Construct projection strategy with explicit lnRNA
   and RT CDS slots.
7. Project only the two anchors through Construct and record
   representability.
8. Generate the three declared view records for those anchors.
9. Materialize or dry-run Infer feature aliases for the three views.
10. Materialize a minimal LatentDNA review workspace or equivalent table-first
    analysis surface from the Infer sidecars.
11. Compute anchor distance metrics and confirm the metrics code refuses
    mismatched views.
12. Emit a triage-readiness report that says either "representation contract
    can proceed to a larger cohort" or "blocked until anchors or constants are
    corrected."

Exit criteria for the vertical slice:

- retron26 and retron43 candidate rows validate;
- both rows have RT and lnRNA provenance;
- both rows either materialize construct views or fail with explicit
  non-representability reasons;
- `AbundancePriorOverlay`, `InferFeatureAlias`, and
  `SpongingAssayObservation` are separate tables or sidecars;
- no candidate schema requires `perturbation_class`,
  `cloning_constraint_set`, or `representation_result`.

### Appendix F. Payload Program Boundary

The v1 payload program is `tetO_sponging_v1`. This keeps payload semantics from
becoming a second uncontrolled search space while the main representation
question is still unresolved.

Payload rules for v1:

- Treat TetO as a fixed or narrowly parameterized payload program.
- Do not rank broad payload variants by construct embeddings before showing
  that retron26/retron43 and abundance overlays have usable geometry.
- Record payload sequence, length, GC, and repeat burden as confound features.
- If a payload edit is included, state whether it is testing payload binding
  logic or retron production robustness; do not let those become the same
  hypothesis.
- Keep bifunctional sponging as a future `payload_program_id`, not a new core
  construct ontology.

The durable machinery question is: does the RT-lnRNA design produce abundant
programmable multicopy ssDNA in the synthetic dual-cassette context? Payload
programs define what that ssDNA is meant to bind after it is produced.

### Appendix G. Analysis Report Contract

The pre-OPAL analysis report should be reproducible from tables, not from a
notebook-only visual interpretation.

Required tables:

| Artifact | Required columns |
| --- | --- |
| `candidate_geometry_metrics.tsv` | `candidate_id`, `construct_context_view_id`, `d_to_retron26_working`, `d_to_retron43_failed`, `working_failed_axis_coordinate`, `off_axis_distance`, `anchor_metric_status` |
| `overlay_regime_metrics.tsv` | `reference_overlay_id`, `regime`, `construct_context_view_id`, `metric_name`, `metric_value`, `n_rows`, `censoring_policy`, `notes` |
| `confound_audit.tsv` | `candidate_id`, `construct_context_view_id`, `length_nt_construct`, `length_nt_lnrna`, `gc_fraction_construct`, `gc_fraction_lnrna`, `repeat_burden`, `inverted_repeat_burden`, `stem_length_estimate`, `codon_policy`, `payload_program_id` |
| `representability_failures.tsv` | `candidate_id`, `construct_contract`, `representation_contract`, `failure_reason`, `blocking_field`, `recommended_action` |
| `wet_lab_candidate_handoff.tsv` | `candidate_id`, `candidate_rationale`, `anchor_role`, `view_id_used_for_rationale`, `required_control`, `expected_readout`, `known_risk` |

The decision report should answer:

- Did retron26 and retron43 separate in each primary view?
- Did the diagnostic lnRNA-span view agree or reveal a likely payload/lnRNA
  artifact?
- Did Crawford abundance bins show local structure in the Eco1 regime?
- Did Khan high producers remain interpretable after projection, or were they
  dominated by source/clade/length effects?
- Did confounds explain the primary geometry?
- Which candidates are selected, and what biological hypothesis does each test?
- Which candidates are rejected or held because the representation contract is
  not trustworthy?

### Appendix H. Risk Register

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Anchor collapse | If retron26 and retron43 are not separated, the view cannot support triage. | Gate all broader analysis on anchor non-collapse in the primary views. |
| Length or repeat confounding | The model may separate rows by obvious sequence mechanics rather than abundance-relevant biology. | Include length, GC, repeat, inverted-repeat, stem-length, and payload confound audits before selection. |
| Payload semantic noise | Payload changes can move embeddings for reasons unrelated to multicopy ssDNA production. | Keep `tetO_sponging_v1` narrow; treat bifunctional payloads as future payload programs. |
| Source-regime overmixing | Khan and Crawford measure different regimes and should not be one label. | Keep separate `reference_overlay_id`, `regime`, and overlay-specific metrics. |
| Non-Eco1 construct incompatibility | Cross-retron RT/lnRNA pairs may exceed 1,600 bp or lack required sequence fields. | Require construct representability before candidate promotion. |
| RT DMS sprawl | RT mutation scans can turn the project into a protein-engineering effort. | Keep RT variants as controls unless a later phase explicitly opens an RT engineering lane. |
| Premature OPAL learning | OPAL cannot learn TF-sponging before true labels exist. | Validate that OPAL `Y` comes only from `SpongingAssayObservation`. |
| Construct materialization gap | Source constants and slot offsets can be known while realized context views are still absent. | Gate Infer/LatentDNA/OPAL on materialized Construct views with audited slot spans. |
| Silent source drift | Processed source tables can change as `dnadesign-data` improves. | Store source paths, row counts, source hashes if available, and verification date in `datasets.yaml`. |
