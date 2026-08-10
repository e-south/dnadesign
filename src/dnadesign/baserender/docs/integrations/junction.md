---
doc_id: baserender-junction-integration
title: junction review integration
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-10
---

# Review a junction design with BaseRender

**Type:** route

**Plane:** downstream-tool

**Owner-boundary:** baserender

**Entry artifact:** verified `junction` `views/three_way_junction_review.v1.json` and an explicit render job
**Exit artifact:** create-only BaseRender review bundle

Use this route after `junction verify` when you need a visual check of selected
targets, fragments, or three-way interfaces. `junction` owns the design and its
evidence. BaseRender only draws the saved record and publishes a separate,
create-only review bundle.

| Input | `views/three_way_junction_review.v1.json` from a verified `junction` bundle |
| --- | --- |
| Adapter | `three_way_junction_review_v1` |
| Render contract | `three_way_junction_review_render_v1` |
| Output | Private BaseRender bundle with one SVG per selected target and a manifest |

The review JSON contains complete sequences and is treated as private. Output
directories use mode `0700`; files use mode `0600`.

## Choose the view that answers the question

| Question | Renderer | Options |
| --- | --- | --- |
| Which fragment oligos are expected to anneal? | `junction_annealed_fragments` | Optional `fragment_ids`; required when a target has more than 18 fragments. |
| How does one target move from separate oligos to the modeled three-way state and recovered duplex? | `junction_three_way_assembly` | `view: assembly` |
| Does a specific interface have the expected three-arm geometry? | `junction_three_way_assembly` | `view: junction_detail` and one to eight `junction_ids` |

Target selection uses BaseRender's normal selection CSV. Fragment and junction
selection belongs in `render.options`; it is not visual style. Unknown options,
unknown IDs, duplicate IDs, and unsafe canvas sizes fail before figure
allocation. A detail view also rejects any junction that would require more
than 512 base glyphs; exact longer sequences remain available in the source
record without forcing Matplotlib to allocate an unbounded number of artists.
The process view accepts at most 64 fragments and a 1,024 bp expected recovered
duplex. Larger assemblies remain available as typed records and can use
selected detail views; BaseRender does not allocate a whole-product molecular
canvas for them.

The fragment map prints exact bases, physical oligo lengths, strand ends,
junction spans, and declared Watson–Crick edges on one nucleotide scale. The
process view keeps both strands visible in the separate-oligo, pre-ligation,
and expected-recovery states; its recovered duplex includes the exact primer
extensions. The junction detail uses the same base spacing horizontally and
vertically, centers a perpendicular barcode helix on the target helix, and
marks the complement-strand nick. It shows `t/t*` and `b/b*` on one shared
three-arm node rather than as unrelated duplex rows.

These are sequence-derived schematics. They do not claim predicted secondary
structure, thermodynamic stability, successful annealing or ligation, PCR
performance, yield, fidelity, or experimental validation.

## Verify the source first

```bash
uv run junction verify review-root/verified-design --format json
```

BaseRender validates the typed review rows, but it does not verify the source
bundle's manifest. Keep that step explicit.

## Example: selected junction details

Keep the job, source bundle, and output under one review root:

```text
review-root/
├── detail.job.yaml
├── verified-design/
│   └── views/
│       └── three_way_junction_review.v1.json
└── reviews/
```

```yaml
version: 4
contract:
  kind: three_way_junction_review_render_v1
bundle:
  path: reviews/junction-detail
input:
  kind: json
  path: verified-design/views/three_way_junction_review.v1.json
  adapter:
    kind: three_way_junction_review_v1
  alphabet: DNA
selection:
  path: selected-targets.csv
  match_on: id
  column: target_id
  on_missing: error
render:
  renderer: junction_three_way_assembly
  options:
    view: junction_detail
    junction_ids:
      - target-a:junction-0001
      - target-a:junction-0002
  style:
    preset: null
    overrides: {}
outputs:
  - kind: images
    dir: images
    fmt: svg
run:
  strict: true
  fail_on_skips: true
```

`selected-targets.csv` contains one target ID:

```csv
target_id
target-a
```

Validate, then run:

```bash
uv run baserender job validate review-root/detail.job.yaml
uv run baserender job run review-root/detail.job.yaml
```

The job writes a new bundle atomically and fails if `bundle.path` already
exists. It never edits the `junction` bundle. The adapter supports image
directories only; video and a combined single-file output are rejected.

The checked-in [three-fragment example](../../../junction/examples/three-fragment-review/)
contains separate jobs for fragment annealing, the assembly process, and junction
details.

## Contract limits

The shared `dnadesign.contracts.visual.ThreeWayJunctionReviewV1` model validates
target reconstruction, fragment order, adjacent junction identities, `t/t*`
and `b/b*` complements, recovery-primer evidence, and document-wide search
receipts. Unknown fields fail validation. `thermodynamic_screening` is fixed at
`not_run` because the current `junction` search is string based.

BaseRender rejects review JSON or a selection CSV above 64 MiB. A job accepts
at most 2,000 review rows, 2,000 selection rows, and 10,000,000 target bases.
The complete source is checked before `input.limit` or selection is applied.

## Method and attribution

The terms follow the three-way-junction and pooled-recovery methods described
by Robinson *et al.* in
[the Sidewinder paper](https://doi.org/10.1038/s41586-025-10006-0) and
[the pooled extension](https://doi.org/10.64898/2026.05.01.722326).
The BaseRender views use an original QA layout; they do not reproduce the
papers' figures or transfer the papers' experimental results to a generated
design.

See `junction`'s [method reference](../../../junction/docs/reference/method-v1.md)
and [sources and scope](../../../junction/docs/reference/sources.md) for the
implementation boundary and unresolved validation gaps.

## Ownership

- `junction` emits the neutral, typed review evidence.
- BaseRender owns deterministic plotting and its output manifest.
- Studies own interpretation, rankings, objectives, and campaign state.
- Review images are advisory. They are not part of `junction` plan identity,
  offline verification, ordering, or laboratory validation.
