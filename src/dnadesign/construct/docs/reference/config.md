## Construct config reference

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-24

### Job shape

One construct job either realizes one template against one input dataset
selection or runs one annotation-aware `normalize_anchor` pass, and writes into
one output dataset. Template-realization jobs may bind one or more named
`input_field` parts from each selected input row into that template.

```yaml
job:
  id: anchor_template_slot_a_window_1kb
  input:
    source:
      kind: usr
      dataset: anchor_parts_demo
      root: outputs/usr_datasets
    field: sequence
    ids: [OPTIONAL_RECORD_ID]
  template:
    id: template_backbone_dual_slot
    source:
      kind: usr
      dataset: template_parts_demo
      root: outputs/usr_datasets
      record_id: TEMPLATE_RECORD_ID
      field: sequence
    circular: true
  parts:
    - name: anchor
      role: anchor
      sequence:
        source: input_field
        field: sequence
      placement:
        kind: replace
        orientation: forward
        locator:
          kind: coordinates
          start: 2300
          end: 2335
        guards:
          replaced_sequence: TTTACGGCTAGCTCAGTCCTAGGTACTATGCTAGC
  realize:
    mode: window
    focal_part: anchor
    required_slots: [anchor]
    window:
      semantics: fixed_total
      reference: center
      direction: symmetric
      size_bp: 1000
      offset_bp: 0
  output:
    target:
      kind: usr
      dataset: anchor_template_slot_a_window_1kb_demo
      root: outputs/usr_datasets
    on_conflict: error
    allow_same_as_input: false
```

### Input

- `input.source.kind`: current backend selector; today this must be `usr`
- `input.source.dataset`: required USR dataset id
- `input.source.root`: required explicit USR root for construct jobs that read USR datasets
- `input.field`: primary sequence-bearing field for one-slot jobs; set this to
  `null` only when every variable `input_field` part declares its own
  `part.sequence.field`
- `input.ids`: optional subset of record ids for selective realization
- construct decides whether a sequence is used as a focal part or as a template; do not encode that role in the USR dataset id itself
- flat `input.dataset`, `input.root`, and scalar `input.source: usr` are rejected; `input.source` must be a mapping

### Template

Supported sources:

- `template.source.kind: usr`: resolve from a USR record
- `template.source.kind: path`: load from a path-backed single-sequence file or single-record FASTA
- `template.source.kind: literal`: inline sequence in config

Fail-fast template rules:

- `template.source.kind=path` rejects multi-record FASTA input
- `template.source.kind=usr` requires `dataset` plus `record_id`
- `template.source.root` defaults to `input.source.root` when omitted
- `template.source.label` is optional provenance text for lineage and preflight summaries; it does not participate in template lookup
- flat `template.kind`, `template.sequence`, `template.path`, `template.dataset`, `template.root`, `template.record_id`, and scalar `template.source` are rejected
- `circular` is explicit in the construct config so window extraction semantics stay audit-visible

### Parts and placement

- each job must include at least one `input_field` part
- each part is a named assembly slot with its own role, sequence source,
  placement, orientation, and guards
- part names must be unique; duplicate names are rejected before placement
  planning so one slot cannot overwrite another slot's resolved site
- multi-slot jobs bind several `input_field` parts from the same input row, for
  example `candidate__lnrna_sequence` and `candidate__rt_cds_sequence`
- every placement now has two explicit sub-blocks:
  - `locator`: how construct finds the site
  - `guards`: optional assertions that prove the resolved site is the intended one
- coordinate locators use `placement.locator.kind=coordinates` with `start` / `end`
- flank locators use `placement.locator.kind=flanks` with `upstream_sequence` / `downstream_sequence`
- coordinate `insert` requires `placement.locator.end == placement.locator.start`
- coordinate `replace` requires `placement.locator.end > placement.locator.start`
- flank `replace` removes everything between the unique upstream and downstream forward-strand flank matches
- flank `insert` is allowed only when the two flanks are adjacent; otherwise construct fails fast
- flank locators that resolve across the template origin are rejected; use explicit coordinates for that case
- placements must not overlap
- zero-width inserts at the same coordinate execute in config order and provenance is recorded in that same execution order
- same-start placements with different template intervals are rejected as ambiguous instead of being silently reordered
- `placement.guards.replaced_sequence` is supported only for `replace` and is strongly recommended for incumbent-swap flows
- `placement.guards.upstream_sequence` / `placement.guards.downstream_sequence` assert the forward-strand flanks around the resolved placement
- `placement.guards.replaced_span_bp` hardens flank-based replacement when you know the intended span length but do not want to repeat the full incumbent sequence
- `placement.guards.require_unique_forward_matches: true` requires every configured guard kmer to match the template exactly once on the forward strand
- flank locators always require unique forward-strand matches for both flanks; repeated flank kmers fail before assembly

### Placement patterns

Common patterns:

- coordinate insert:
  `kind: insert`, `locator.kind: coordinates`, `start == end`, no incumbent interval
- coordinate replace:
  `kind: replace`, `locator.kind: coordinates`, `start < end`, no guards
- incumbent swap:
  `kind: replace` plus `guards.replaced_sequence`
- flank-guarded swap:
  `kind: replace`, `locator.kind: coordinates`, plus one or both guard flanks
- flank-located swap:
  `kind: replace`, `locator.kind: flanks`, optional `guards.replaced_sequence` or `guards.replaced_span_bp`
- fully guarded swap:
  `kind: replace` plus `guards.replaced_sequence`, guard flanks, and `guards.require_unique_forward_matches: true`

### Realization

- `mode: full_construct`: write the entire realized construct
- `mode: window`: extract a focal window around `focal_part`
- `realize.required_slots`: optional list of part names that must remain
  present as one contiguous span in the emitted output; windowed jobs fail
  before write-back if a required slot would be clipped or split
- `realize.window.semantics=fixed_total`: emitted output length is fixed by `size_bp`
- `realize.window.reference=start|center|end`: choose the focal point inside the realized part
- `realize.window.direction=symmetric|five_prime|three_prime`: symmetric is the default; `five_prime` and `three_prime` are resolved relative to part orientation
- `realize.window.semantics=anchor_plus_context`: emitted output spans the full focal part plus explicit `upstream_bp` and `downstream_bp`
- `realize.window.offset_bp`: optional fixed-total shift inside a `fixed_total` window
- `realize.focal_point`, `realize.window_bp`, and `realize.anchor_offset_bp` are rejected; `realize.window` is the only supported window contract
- circular templates support wraparound extraction
- linear templates fail if the requested window would exceed boundaries
- fixed-total windows fail if the focal part itself is longer than the requested emitted size

### Normalize-anchor mode

`job.mode: normalize_anchor` is the analysis-view sibling path for reference/control material.
It derives a separate analysis product from a biological input row; it does not rewrite the
native/reference sequence.

- `job.normalize_anchor.product_kind`: today this must be `analysis_window`
- `job.normalize_anchor.target_length`: required emitted length, for example `60`
- `job.normalize_anchor.focal_selector`: ordered selector chain; the first successful selector wins
- `job.normalize_anchor.over_length_policy.kind=trim`: emit an exact-length window from the parent sequence around the focal point
- `job.normalize_anchor.over_length_policy.window_anchor=upstream_of_focal`: emit the half-open
  upstream interval ending at the focal offset, for sources that declare a fixed sequence-relative
  TSS or other feature offset
- `job.normalize_anchor.under_length_policy.kind=expand_from_template`: expand a short parent sequence only from an explicit template context
- `job.normalize_anchor.feature_retention_policy`: fail/warn rules for retained, clipped, and lost annotated roles
- `job.normalize_anchor.fallback_policy.allow_low_confidence`: opt-in gate for low-confidence selectors such as `sequence_midpoint`
- `job.normalize_anchor.output_sequence_view.create`: emit a USR `_views/sequence_views.parquet` row for the derived product

Supported focal selectors:

- `annotation_pair_midpoint`: require one first feature and one second feature, then use the midpoint
  between their feature centers
- `annotation_feature_center`
- `sequence_offset`: use a declared 0-based sequence-relative offset such as a source-provided TSS
  offset; pair it with `window_anchor=upstream_of_focal` when the intended product excludes the
  focal base
- `sequence_midpoint`

For the promoter-reference core60 lane, the intended high-confidence selector is
`annotation_pair_midpoint` over `sigma70_minus35` and `sigma70_minus10`. A config that only
declares this selector has no midpoint fallback: missing or ambiguous sigma-site annotations fail
before write-back.

Over-length inputs use the annotation-pair focal point to choose an exact 60 bp analysis window.
Construct evaluates candidate windows that contain the focal point, prefers retention of the
selected sigma-site features, records clipped/lost annotations in the `derived` overlay, and writes
`derived__source_interval_start_0` / `derived__source_interval_end_0` as 0-based half-open parent
coordinates. The emitted row is `analysis_only=true` and `derived__product_kind=analysis_window`.

Under-length inputs are never padded with arbitrary bases. They require
`under_length_policy.kind=expand_from_template`. The study reference lane uses
`placement_ref: replace:<start_0>-<end_0>`: Construct replaces that template interval with the short
input sequence, extracts the exact target-length window around the sigma-site focal point, and records
`derived__added_left_bp`, `derived__added_right_bp`, and template provenance. This is a pDual-context
analysis view, not a new native promoter.

Fail-fast rules for `normalize_anchor`:

- ambiguous annotation matches fail before sequence emission
- malformed `seq_annot__features` payloads fail before selector fallback or
  feature-retention reporting
- `sequence_midpoint` is low-confidence and requires `fallback_policy.allow_low_confidence: true`
- short inputs require `under_length_policy`
- template expansion must still emit the exact target length
- required retained roles are enforced after trimming/expansion and before write-back

### Output variants

Classic template-realization jobs may add `job.output_variants` to emit explicit forward and
whole-output reverse-complement products from one realized forward construct.

- `product_kind: realized_context` with `orientation: forward` emits the forward context product
- `product_kind: realized_context` with `orientation: reverse_complement` emits the whole-output
  reverse-complement context product
- `anchor_part`: optional named part whose emitted span is copied into that
  sequence-view row's `anchor_start_0` / `anchor_end_0` bounds; use this for
  multi-slot studies that need slot-specific `anchor_mean` features
- `anchor_window_size_bp`: optional fixed pooling-window length for
  `anchor_mean` variants with `anchor_part`; Construct centers the window on
  the named slot when possible, clamps it inside the emitted context, and fails
  if the slot cannot fit inside the fixed window
- `view_name`: optional explicit sequence-view name for a study representation,
  for example `lnrna_fixed_256bp_window_in_construct_anchor_mean`
- variant sequence views require an anchor handoff span. For multi-slot jobs
  whose parts are not named or role-tagged `anchor`, set either
  `output_variants[].anchor_part` for the slot-specific view or
  `realize.focal_part` for one primary row-level compatibility span.
- reverse-complement variants carry emitted-orientation anchor bounds plus
  `construct__forward_anchor_start` / `construct__forward_anchor_end` for downstream audits
- reverse-complement variants are the reverse complement of the full emitted context sequence, not
  the reverse complement of a truncated anchor-only sequence
- reverse-complement variants transform `construct__slots[].start` /
  `construct__slots[].end` into emitted reverse-complement coordinates while
  preserving `forward_start` / `forward_end` for audits
- Infer `anchor_mean` consumers should use the emitted-orientation `construct__anchor_start` /
  `construct__anchor_end` or matching sequence-view bounds directly; reverse-complement bounds are
  already transformed by Construct using `L-b, L-a`
- `anchor_mean` is an Infer pooling instruction over those coordinates. Construct still emits the
  full context sequence, and Infer should pass that full sequence through the model before pooling
  the anchor span.
- Construct only defines the emitted sequence and span coordinates. For causal
  Evo2 models, an Infer `anchor_mean` vector averages prefix-conditioned token
  states in that emitted orientation; two-sided context requires a separate
  reverse-complement pass or another explicit downstream representation.
- multiple output variants may point at the same emitted sequence and
  orientation when they declare distinct `anchor_part` / `view_name` pairs;
  Construct writes one base sequence row plus distinct sequence-view rows
- semantic variants may share one base sequence id; construct writes distinct sequence-view rows
  instead of forcing duplicate base records
- with `output.on_conflict=ignore`, already-present base rows are skipped, but planned sequence-view
  rows are still written idempotently so reruns can complete missing semantic views

### Output

- `output.target.kind`: current write backend selector; today this must be `usr`
- `output.target.dataset`: required USR dataset id
- `output.target.root`: explicit output USR root; when omitted it defaults to `input.source.root`
- `output.record_source`: optional human-readable source string written onto emitted USR rows
- `output.on_conflict`:
  - `error`: fail during preflight if any planned output id already exists
  - `ignore`: skip already-present output ids during run
- `output.allow_same_as_input`: defaults to `false`; set to `true` only for intentional recursive accumulation
- flat `output.dataset`, `output.root`, and `output.source` are rejected; use `output.target.*` plus `output.record_source`

### Multi-slot and multi-template studies

The current construct schema is intentionally one-template-per-job, but a
single template job may assemble multiple named slots from one input row. Use
multiple construct projects when a study needs multiple templates, context
lengths, or backbone choices.

Represent larger template matrices by:

1. creating multiple config files
2. inventorying them in `construct.workspace.yaml` under `project.artifacts.config`
3. using `construct workspace show` plus `validate`/`run` per project

This keeps each construct spec auditable and avoids hiding a template matrix
inside one oversized config. Multi-slot placement remains part of the public
job schema, not a study-local precomposition workaround.

Use `construct workspace doctor` to keep the registry and those config files aligned, then use
`construct workspace validate-project` or `construct workspace run-project` to execute by project id.

For the downstream handoff fields that infer reads from templated rows, see
[template-contexts.md](template-contexts.md).
