---
doc_id: study-rt-lnrna-sponging-construct-triage-representation-contract
surface: study-context
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-05-26
---

## Representation Contract

The representation contract is
`dual_cassette_construct_context_embedding_v1`.

The Construct export materializes six source sequence-view names for every
promoted Construct subject:

| View | Required source | Pooling |
| --- | --- | --- |
| `dual_cassette_2000bp_seq_mean` | One forward `realized_context` sequence view with `context_kind=template_custom`. | `seq_mean` over the full 2,000 bp context. |
| `dual_cassette_2000bp_reverse_complement_seq_mean` | One reverse-complement `realized_context` companion row with `context_kind=template_custom`. | `seq_mean` over the full RC context. |
| `lnrna_fixed_384bp_window_in_construct_anchor_mean` | Forward full construct context with a fixed 384 bp pooling window containing the lnRNA slot. | `anchor_mean` over the declared window. |
| `lnrna_fixed_384bp_window_in_construct_reverse_complement_anchor_mean` | Reverse-complement full construct context with the orientation-aware 384 bp lnRNA pooling window. | `anchor_mean` over the declared window. |
| `rt_cds_fixed_1600bp_window_in_construct_anchor_mean` | Forward full construct context with a fixed 1,600 bp pooling window containing the RT CDS slot. | `anchor_mean` over the declared window. |
| `rt_cds_fixed_1600bp_window_in_construct_reverse_complement_anchor_mean` | Reverse-complement full construct context with the orientation-aware 1,600 bp RT CDS pooling window. | `anchor_mean` over the declared window. |

The views are study representation names, not new USR `product_kind` values.
Persist them through sequence-view names, aliases, and view semantics. Infer
must consume declared sequence views and must not reverse-complement, pad,
window, or infer missing spans implicitly.

`src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/infer_readiness.py`
is the executable post-Construct readiness gate. A materialized construct
subject is Infer-ready only when its input envelope carries explicit lnRNA and
RT CDS slot sequences, its consolidated output has exactly one forward and one
reverse-complement 2,000 bp context row, and its output dataset carries exactly
the six source sequence-view names above. Missing, duplicate, or unsupported
view names fail before Infer sequence-view completion or Evo2 execution.

Construct provides raw source views only. It must not formalize a pre-Infer
forward/reverse-complement concat view or attach a `downstream_transform` to the
Construct projection manifest. Infer extracts sidecars independently for the
declared forward and reverse-complement sequence views; LatentDNA owns any
post-inference block-normalized concat after those sidecars exist.

Construct provides the coordinate authority through named slots. Every emitted
context must carry auditable `construct__slots` spans for both `lnrna` and
`rt_cds`. A slot-specific `anchor_mean` view declares the fixed pooling window
that contains the named slot: 384 bp for `lnrna`, 1,600 bp for `rt_cds`. The
lnRNA views map `construct_output_anchor_part: lnrna`, and the RT CDS views
map `construct_output_anchor_part: rt_cds`; the sequence-view
`anchor_start_0` / `anchor_end_0` coordinates are the fixed pooling window, not
the variable-length biological slot. Row-level
`construct__anchor_start` / `construct__anchor_end` is compatibility metadata for
single-span consumers, not the authority for every slot in this dual-cassette
study.

The forward sequence-view row for `dual_cassette_2000bp_seq_mean` and the
reverse-complement sequence-view row for
`dual_cassette_2000bp_reverse_complement_seq_mean` are separate Infer inputs.
Do not duplicate either emitted sequence merely to imply a concat. Derived
bidirectional aliases may reference both post-Infer vectors, but that is a
LatentDNA representation step rather than a Construct sequence-view contract.
The full 2,000 bp context is `2000bp-region.gb`, which maps to pES-retron-26
`[56,2056)` in zero-based half-open vector coordinates. The GenBank catalog
uses one projection path for every representable variant. In the retron26
fixture row the region-relative slot spans are `lnrna: [130,303)` and
`rt_cds: [468,1431)`. In the retron43 fixture row the longer lnRNA shifts the
emitted window start to 63, giving `lnrna: [123,310)` and
`rt_cds: [475,1438)`; the inter-slot distance stays constant while outer
flanks absorb the length delta. The same policy applies to RT length changes:
Sso7d-fusion RT rows keep the full RT slot inside the 2,000 bp view by trimming
outer flanks. The policy never clips the lnRNA or RT slot itself.
The bidirectional anchor views are also downstream derived vector views, not
extra sequence rows: LatentDNA concatenates the declared forward and RC anchor
aliases after Infer writes sidecars.

RT-lnRNA Infer configs must select these inputs by explicit `view_name`.
Selecting only by `product_kind=realized_context` and orientation is ambiguous
because whole-context, lnRNA-anchor, and RT-anchor views share those fields.

The fixed-size representation table contract lives at
`../operations/contract/schemas/representation-table.schema.yaml`. It declares
the six source views, expected Evo2 7B sidecar outputs, fixed-size vector export
dimensions, OPAL `X` selection posture, and Khan/Crawford source-overlay
integration rules.

### LatentDNA Surface

The LatentDNA workspace at
`../../../../src/dnadesign/latentdna/workspaces/rt_lnrna_sponging_construct_triage/config.yaml`
is a planned gallery contract until Evo2 sidecars exist. It declares, for each
of the six source sequence-view selectors:

- one Evo2 7B `intermediate_embedding` vector sidecar;
- one Evo2 7B `output_layer_mean` vector sidecar;
- one `log_likelihood__total` scalar sidecar;
- one `log_likelihood__mean_per_token` scalar sidecar.

The browser/gallery surface has eight derived panels: four intermediate views
and four output-layer companions. The four intermediate views are the full
2,000 bp forward/RC concat, lnRNA-slot bidirectional concat, RT CDS-slot
bidirectional concat, and lnRNA plus RT slot-pair concat. The output-layer
companions use the same view geometry as diagnostics; they do not automatically
become OPAL `X`.

The planned plot surface has four study-agnostic pieces ported from the
promoter-style LatentDNA work:

- a representation-health metric panel that gates obvious collapse using PCA
  effective rank, PC1 variance concentration, and pairwise-distance spread;
- an appendix PCA scree diagnostic for the same eight intermediate and
  output-layer views;
- a Khan/Crawford ordinal overlay audit, with separate axes for Khan RT-DNA
  abundance priors and Crawford Eco1 msDNA abundance priors;
- an appendix UMAP gallery spanning intermediate and output-layer views with
  construct and overlay hue controls.

Khan and Crawford overlay columns preserve `raw_value` and `normalized_value`
as source-scoped numeric fields. They are not on one shared numeric scale.
`ordinal_bin` is a secondary metadata axis for source-local geometry review,
not a replacement label and not OPAL `Y`. Crawford design-reference rows and
abundance-observation rows remain distinct source record types; Construct
promotion moves forward only with abundance-affiliated lnRNA sequences, while
reference-only sequences are retained as source provenance and issue records.

Khan and Crawford rows can participate in LatentDNA overlays only after they
become construct-compatible construct subject rows with explicit `construct_subject__lnrna_sequence`
and `construct_subject__rt_cds_sequence` authority. Promoted rows must enter
`rt_lnrna_sponging_construct_triage_construct_slot_inputs_v1`, be emitted into
the consolidated
`rt_lnrna_sponging_construct_triage_construct_contexts_2000bp_v1` dataset, and
receive the same six source sequence-view declarations as the retron26/retron43
examples before Infer runs. Overlay-only rows must not be passed to Infer.

### Failure Rules

- Missing constants fail before Infer.
- Invalid DNA alphabet fails before Infer.
- Any focal lnRNA span, lnRNA slot, or RT CDS slot clipping fails before Infer.
  Prefix/suffix flank truncation or extension is allowed only when the catalog
  can still emit full lnRNA and RT slot spans inside the 2,000 bp context.
- Missing reverse-complement source view blocks the downstream bidirectional
  derived aliases.
- Missing lnRNA or RT CDS slot span fails the matching anchor-mean view.
- Missing construct-subject bridge metadata on output rows, missing explicit
  `view_name`, or fewer/more than the six required source sequence views fails
  the materializer postcondition before Infer runs.

The slot anchor views run Evo2 over the full construct context and pool the
declared fixed window that contains the slot span. They are not naked
lnRNA-only or RT-only embeddings.
P4/foldback/stem views remain appendix-only until construct-subject-owned subspan
coordinates are explicit and source-backed.
