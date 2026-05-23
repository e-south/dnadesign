---
doc_id: study-rt-lnrna-sponging-construct-triage-representation-contract
surface: study-context
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-05-23
---

## Representation Contract

The representation contract is
`dual_cassette_construct_context_embedding_v1`.

The Phase 2 Construct export materializes six source sequence-view names for
the two controls:

| View | Required source | Pooling |
| --- | --- | --- |
| `dual_cassette_1600bp_seq_mean` | One forward `realized_context` sequence view with `context_kind=template_custom`. | `seq_mean` over the full 1,600 bp context. |
| `dual_cassette_1600bp_fwd_rc_concat` | One reverse-complement `realized_context` companion row with `context_kind=template_custom`. | `seq_mean` over the full RC context; downstream concat pairs it with the forward row. |
| `lnrna_span_in_construct_anchor_mean` | Forward full construct context with lnRNA span bounds. | `anchor_mean` over the `lnrna` slot. |
| `lnrna_span_in_construct_reverse_complement_anchor_mean` | Reverse-complement full construct context with orientation-aware lnRNA span bounds. | `anchor_mean` over the `lnrna` slot. |
| `rt_cds_span_in_construct_anchor_mean` | Forward full construct context with RT CDS span bounds. | `anchor_mean` over the `rt_cds` slot. |
| `rt_cds_span_in_construct_reverse_complement_anchor_mean` | Reverse-complement full construct context with orientation-aware RT CDS span bounds. | `anchor_mean` over the `rt_cds` slot. |

The views are study representation names, not new USR `product_kind` values.
Persist them through sequence-view names, aliases, and view semantics. Infer
must consume declared sequence views and must not reverse-complement, pad,
window, or infer missing spans implicitly.

Construct provides the coordinate authority through named slots. Every emitted
context must carry auditable `construct__slots` spans for both `lnrna` and
`rt_cds`. A slot-specific `anchor_mean` view must also declare the slot that
supplies its sequence-view `anchor_start_0` / `anchor_end_0` bounds; the
lnRNA views map `construct_output_anchor_part: lnrna`, and the RT CDS views
map `construct_output_anchor_part: rt_cds`. Row-level
`construct__anchor_start` / `construct__anchor_end` is compatibility metadata for
single-span consumers, not the authority for every slot in this dual-cassette
study.

The forward sequence-view row for `dual_cassette_1600bp_seq_mean` is the forward
member of the forward/reverse-complement concat contract. Do not duplicate the
same forward emitted sequence merely to attach a second view name; the concat
feature layer should reference that forward row plus the declared
reverse-complement row.
The full 1,600 bp context is `1600bp-region.gb`, which maps to pES-retron-26
`[56,1656)` in zero-based half-open vector coordinates. In the retron26 control
row the region-relative anchors are `lnrna: [130,303)` and `rt_cds: [468,1431)`.
Retron43's 14 bp longer lnRNA shifts the emitted window start to 63, so the
region-relative anchors become `lnrna: [123,310)` and `rt_cds: [475,1438)`;
the 165 bp interstitial remains constant while the prefix/suffix flanks are
trimmed symmetrically.
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
1,600 bp forward/RC concat, lnRNA-slot bidirectional concat, RT CDS-slot
bidirectional concat, and lnRNA plus RT slot-pair concat. The output-layer
companions use the same view geometry as diagnostics; they do not automatically
become OPAL `X`.

The planned plot surface has three study-agnostic pieces ported from the
promoter-style LatentDNA work:

- a representation-health metric panel;
- a Khan/Crawford ordinal overlay audit, with separate axes for Khan RT-DNA
  abundance priors and Crawford Eco1 msDNA abundance priors;
- an appendix UMAP gallery spanning intermediate and output-layer views with
  construct and overlay hue controls.

Khan and Crawford overlay columns are declared as planned metadata until the
overlay resolver attaches real rows. Preserve Khan and Crawford `raw_value` and
`normalized_value` as the numeric source fields. `ordinal_bin` is a secondary
metadata axis for geometry review, not a replacement label and not OPAL `Y`.
Crawford design-reference rows remain a separate sequence/design reference
lane; they are not abundance observations.

### Failure Rules

- Missing constants fail before Infer.
- Invalid DNA alphabet fails before Infer.
- Any focal lnRNA span, lnRNA slot, or RT CDS slot clipping or truncation fails
  before Infer.
- Missing reverse-complement view fails the concat alias.
- Missing lnRNA or RT CDS slot span fails the matching anchor-mean view.

The slot anchor views run Evo2 over the full construct context and pool the
declared slot span. They are not naked lnRNA-only or RT-only embeddings.
P4/foldback/stem views remain appendix-only until candidate-owned subspan
coordinates are explicit and source-backed.
