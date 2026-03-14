## Construct Development Journal

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-14

This journal tracks `dnadesign.construct` design and implementation decisions, scope boundaries, and validation notes.

## 2026-03-14 - Phase 0 Kickoff (Create-Plan + MVP Spec)

### Semantic objective

`construct` is the sibling tool that realizes new DNA sequences from:

- anchor sequences from USR
- template context such as plasmids or other backbone sequences
- explicit placement and windowing rules

The realized sequences are first-class USR records. The realization provenance is stored as `construct__*` lineage metadata.

### Plan intent summary

Ship a real first implementation of `construct` as a sibling tool with a layout consistent with other dnadesign packages and a usable MVP workflow, not a placeholder scaffold.

### Explicit scope

- In scope:
  - new `construct` package with lightweight top-level surface
  - config-driven run path from USR anchors to derived USR records
  - template loading from literal sequence or file
  - multi-part placement on a shared template, with first-class support for one or many placed parts
  - windowed extraction around a focal part, plus full-construct output mode
  - construct lineage overlays in USR
  - CLI validate and workspace scaffolding
  - targeted tests for package layout, config validation, workspace behavior, and runtime realization
- Out of scope:
  - genbank parsing, rich plasmid annotation import, or graphical map manipulation
  - generalized edit history or resume/retry orchestration
  - infer-specific shortcuts or cross-tool internal imports

### Boundary and contract decisions

- `construct` owns realization logic only.
- `usr` remains the canonical home for resulting sequence records.
- `infer` remains downstream and consumes construct outputs as ordinary USR sequences.
- `construct` must use public USR APIs plus the documented `registry.yaml` artifact contract, not internal `usr.src.*` imports.
- Missing template files, missing input fields, overlapping placements, invalid window bounds, and duplicate realized IDs must fail fast.

### MVP config contract

```yaml
job:
  id: promoter_context_1kb
  input:
    source: usr
    dataset: densegen_anchor_set
    root: outputs/usr_datasets
    field: sequence
  template:
    id: plasmid_demo
    path: inputs/template.fa
    circular: true
    source: inputs/template.fa
  parts:
    - name: anchor
      role: anchor
      sequence:
        source: input_field
        field: sequence
      placement:
        kind: replace
        start: 100
        end: 160
        orientation: forward
        expected_template_sequence: REPLACE_WITH_TEMPLATE_INTERVAL
  realize:
    mode: window
    focal_part: anchor
    focal_point: center
    anchor_offset_bp: 0
    window_bp: 1000
  output:
    dataset: densegen_anchor_set_construct_1kb
    root: outputs/usr_datasets
```

### Ordered action checklist

1. Create the new sibling-tool layout:
   - top README
   - docs index surfaces
   - dev journal
   - `src/`, `tests/`, and `workspaces/`
2. Add package wiring and CLI entrypoints:
   - `construct run`
   - `construct validate config`
   - `construct workspace where|init`
3. Implement the realization runtime:
   - load config
   - read anchor rows from USR
   - load template sequence
   - apply ordered part placements
   - extract either a focal window or the full construct
4. Persist outputs to USR:
   - create output dataset if needed
   - import realized sequences as base records
   - attach `construct__*` lineage overlay columns
5. Add tests and validate:
   - package layout contracts
   - config validation behavior
   - workspace scaffold behavior
   - runtime realization for linear and circular cases

### Validation and risk handling

- Required checks:
  - targeted `pytest` for new construct tests
  - `ruff check` on touched construct files
- Risk controls:
  - strict error behavior only; no hidden fallback to alternate paths or sequence fields
  - overlap and duplicate realized-sequence collisions are hard errors
  - registry bootstrap is explicit and deterministic for the output USR root

### Open questions for later phases

- When richer plasmid-map sources arrive, decide whether to add a dedicated parser layer or keep maps external to the first construct contract.
- Decide whether future multi-part constructs should support cross-boundary replacement spans on circular templates or keep first-class wrap only at the extraction stage.

## 2026-03-14 - Phase 1 Tracer Bullet (Promoter Swap + Control Anchors)

### Objective

Harden `construct` around the first real biological use case:

- control/wild-type promoter anchors that live in USR beside DenseGen-derived anchors
- a circular plasmid/scaffold template that contains an incumbent promoter interval
- coordinate-driven replacement of that incumbent interval
- fixed-length realized windows for downstream `infer`

### Contract refinements

- Placement semantics are explicit:
  - `kind: insert` requires `start == end`
  - `kind: replace` requires `end > start`
- Replacement specs may declare `expected_template_sequence`.
  - Runtime verifies the template interval before constructing outputs.
  - This keeps the contract coordinate-based while still protecting against wrong-incumbent swaps.
- `construct validate config --config <path> --runtime` now resolves the template, input dataset, and projected output lengths without writing USR data.

### Control-anchor dataset stance

- Control promoters and wild-type promoters are ordinary USR anchor records, not DenseGen exceptions.
- The workspace scaffold now includes `inputs/anchor_manifest.template.yaml` as a worksheet for:
  - `spyP_MG1655`
  - `sulAp`
  - `soxS`
  - `J23105`
  - optional `pDual-10` template reference metadata

### Fidelity rule for pasted plasmid sequence

- Long plasmid DNA pasted into chat is not treated as canonical tracked sequence content.
- The workspace scaffold keeps placeholders and notes instead of committing `pDual-10` sequence text from chat.
- Real runs should use a canonical FASTA file and, when possible, a checksum recorded in the workspace manifest.

### Tracer-bullet example assumptions captured from discussion

- The first 1 kb promoter-context example is a scaffold-before-insertion template.
- The incumbent promoter interval is replaced, not substring-discovered at runtime.
- The discussion-derived incumbent `J23105` interval on the 1 kb snippet is zero-based half-open `[405, 440)`.
- Downstream fixed-length context windows should treat `pDual-10` as circular.
