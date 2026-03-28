## Construct config reference

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-24

### Job shape

One construct job realizes one template against one input dataset selection and writes into one output dataset.

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
- `input.field`: sequence-bearing field for `input_field` parts
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
- `realize.window.semantics=fixed_total`: emitted output length is fixed by `size_bp`
- `realize.window.reference=start|center|end`: choose the focal point inside the realized part
- `realize.window.direction=symmetric|five_prime|three_prime`: symmetric is the default; `five_prime` and `three_prime` are resolved relative to part orientation
- `realize.window.semantics=anchor_plus_context`: emitted output spans the full focal part plus explicit `upstream_bp` and `downstream_bp`
- `realize.window.offset_bp`: optional fixed-total shift inside a `fixed_total` window
- `realize.focal_point`, `realize.window_bp`, and `realize.anchor_offset_bp` are rejected; `realize.window` is the only supported window contract
- circular templates support wraparound extraction
- linear templates fail if the requested window would exceed boundaries
- fixed-total windows fail if the focal part itself is longer than the requested emitted size

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

### Multi-template and matrix studies

The current construct schema is intentionally one-template-per-job. Represent larger studies by:

1. creating multiple config files
2. inventorying them in `construct.workspace.yaml` under `project.artifacts.config`
3. using `construct workspace show` plus `validate`/`run` per project

This keeps each construct spec auditable and avoids hiding a job matrix inside one oversized config.

Use `construct workspace doctor` to keep the registry and those config files aligned, then use
`construct workspace validate-project` or `construct workspace run-project` to execute by project id.

For the downstream handoff fields that infer reads from templated rows, see
[template-contexts.md](template-contexts.md).
