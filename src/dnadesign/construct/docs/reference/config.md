## construct config reference

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-14

### Job shape

One construct job realizes one template against one input dataset selection and writes into one output dataset.

```yaml
job:
  id: promoter_swap_slot_a_window_1kb
  input:
    source: usr
    dataset: mg1655_promoters
    root: outputs/usr_datasets
    field: sequence
    ids: [OPTIONAL_RECORD_ID]
  template:
    id: pDual-10
    kind: usr
    dataset: plasmids
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
        start: 2300
        end: 2335
        orientation: forward
        expected_template_sequence: TTTACGGCTAGCTCAGTCCTAGGTACTATGCTAGC
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
    dataset: pdual10_slot_a_window_1kb_demo
    root: outputs/usr_datasets
    on_conflict: error
    allow_same_as_input: false
```

### Input

- `input.dataset`: required USR dataset id
- `input.root`: required explicit USR root for construct jobs that read USR datasets
- `input.field`: sequence-bearing field for `input_field` parts
- `input.ids`: optional subset of record ids for selective realization
- construct decides whether a sequence is used as a focal part or as a template; do not encode that role in the USR dataset id itself

### Template

Supported sources:

- `kind: usr`: resolve from a USR record
- `kind: path`: load from a path-backed single-sequence file or single-record FASTA
- `kind: literal`: inline sequence in config

Fail-fast template rules:

- `kind: path` rejects multi-record FASTA input
- `kind: usr` requires `dataset` plus `record_id`
- `template.root` defaults to `input.root` when omitted
- `circular` is explicit in the construct config so window extraction semantics stay audit-visible

### Parts and placement

- each job must include at least one `input_field` part
- `insert` requires `start == end`
- `replace` requires `end > start`
- placements must not overlap
- zero-width inserts at the same coordinate execute in config order and provenance is recorded in that same execution order
- same-start placements with different template intervals are rejected as ambiguous instead of being silently reordered
- `expected_template_sequence` is supported only for `replace` and is strongly recommended for incumbent-swap flows

### Realization

- `mode: full_construct`: write the entire realized construct
- `mode: window`: extract a focal window around `focal_part`
- `realize.window.semantics=fixed_total`: emitted output length is fixed by `size_bp`
- `realize.window.reference=start|center|end`: choose the focal point inside the realized part
- `realize.window.direction=symmetric|five_prime|three_prime`: symmetric is the default; `five_prime` and `three_prime` are resolved relative to part orientation
- `realize.window.semantics=anchor_plus_context`: emitted output spans the full focal part plus explicit `upstream_bp` and `downstream_bp`
- `realize.window.offset_bp`: optional fixed-total shift for compatibility with older centered-window configs
- legacy `focal_point`, `window_bp`, and `anchor_offset_bp` are still accepted as a compatibility path, but new configs should prefer `realize.window`
- circular templates support wraparound extraction
- linear templates fail if the requested window would exceed boundaries
- fixed-total windows fail if the focal part itself is longer than the requested emitted size

### Output

- `output.dataset`: required USR dataset id
- `output.root`: explicit output USR root; when omitted it defaults to `input.root`
- `output.on_conflict`:
  - `error`: fail during preflight if any planned output id already exists
  - `ignore`: skip already-present output ids during run
- `output.allow_same_as_input`: defaults to `false`; set to `true` only for intentional recursive accumulation

### Multi-template and matrix studies

The current construct schema is intentionally one-template-per-job. Represent larger studies by:

1. creating multiple config files
2. inventorying them in `construct.workspace.yaml`
3. using `construct workspace show` plus `validate`/`run` per project

This keeps each construct spec auditable and avoids hiding a job matrix inside one oversized config.

Use `construct workspace doctor` to keep the registry and those config files aligned, then use
`construct workspace validate-project` or `construct workspace run-project` to execute by project id.

For the downstream handoff fields that infer reads from templated rows, see
[template-contexts.md](template-contexts.md).
