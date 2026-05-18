## Generic Linear ssDNA Composition Dev Spec

This is the detailed source spec and design history. Start with the
[linear ssDNA composition entry point](2026-05-13-generic-linear-ssdna-composition.md)
unless you need the full rationale.

**Status:** accepted implementation reference
**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-14
**Primary study:** `retron_hairpin_design`
**Current generic authority:** [Construct linear ssDNA composition reference](../../../../../src/dnadesign/construct/docs/reference/linear-ssdna-composition.md)
**Architecture decision:** [ADR 0002](../../../../architecture/decisions/adr-0002-generic-linear-ssdna-composition.md)
**Study handoff:** [Retron linear ssDNA composition handoff](../../../../studies/retron_hairpin_design/contexts/linear-ssdna-composition.md)
**Implementation record:** [Generic linear ssDNA composition](../../../../exec-plans/completed/2026-05-13-generic-linear-ssdna-composition.md)
**Follow-up plan:** [Linear ssDNA composition hardening follow-ups](../../../../exec-plans/active/2026-05-14-linear-ssdna-composition-hardening-followups.md)

### Executive Summary

This spec is an accepted implementation reference and design history. Use the
Construct reference doc above for current generic procedure and bundle-routing
authority; use the Retron handoff only for Retron-specific study choices.

Add a contract-first, generic linear ssDNA composition capability to
`Construct` so it can assemble multicopy linear single-stranded DNA products
from modular sequence components. The dogfooding case is the Retron/TetO
hairpin workflow, but the implementation must remain Retron-agnostic.

The current Retron product posture has a study-owned layer above the generic
composition/folding stack: users provide or select primitive parts through a
lab-facing MSD shorthand label, the Retron study validates and freezes that
selection into `msd_design_reference_v1` / `msd_design_catalog_v1`, and only
then do Construct/Folding/BaseRender act as service/rendering surfaces for
assembly or visualization. This ID-to-reference layer is not a generic
top-level tool and must not create one stale workspace per requested design.

The central ontology is:

```text
Segments assemble the sequence; annotations interpret spans.
```

For the Retron/TetO case, Construct should concatenate ordered physical
segments such as 5-prime flank, payload, snapback/cap segment,
reverse-complement payload arm, and 3-prime flank. It should then emit
annotation spans for meanings such as `stem_base_left`, `payload_primary`,
`snapback_foldback_geometry`, `payload_complement`, `stem_base_right`, `TetO`,
`scar`, or `cap_subregion`.

Construct validates sequence mechanics, span coverage, transforms, repeats,
provenance, and topology. It does not know what `TetO`, `TetR`,
`retron-43`, `scar_nick`, or `Snapback route` biologically mean.
Publication-facing labels, component hues, and Retron-specific plot titles are
configuration data under `visual.display_profile`; generic Construct and
Folding code must consume that profile rather than hard-coding study terms.

The implementation introduces three durable contract surfaces:

1. `linear_ssdna_composition_v1`: ordered segment assembly, annotations,
   repeats, transforms, provenance, and validation.
2. `secondary_structure_prediction_v1`: backend-neutral folding result for
   the ViennaRNA Package or another secondary-structure engine. In this spec,
   `RNAfold` means the ViennaRNA command-line program, not a separate parent
   backend.
3. Visual contracts, reusing `sequence_evidence_map_v1` with explicit scope:
   one canonical component-unit evidence map for BaseRender and
   ViennaRNA-native annotation. Repeat-expanded products must not be used as
   visual or folding evidence unless a future contract explicitly introduces a
   separate repeat-expanded analysis surface.

Ownership boundaries are fixed:

| Domain | Owner |
| --- | --- |
| Generic linear ssDNA composition | `Construct` |
| Snapback/released-product geometry candidates | `Cruncher.snapback` |
| Type IIS scar plus terminal nick feasibility | `Cruncher.scar_nick` |
| Retron-specific selection/rationale | study records, and only later study-owned selector code |
| Retron MSD shorthand parsing and design-reference catalogs | `dnadesign.studies.studies.retron_hairpin_design` plus study registry |
| Folding execution | separate folding/backend layer |
| Rendering | `BaseRender`, consuming contracts only |
| Durable sequence persistence | local artifact bundle first; optional USR later |

One implementation caveat: the literal retron-43 sequence and earlier shorthand
spans have a coordinate/length tension. The ontology is settled, but the
implementation must standardize on a coordinate convention and compute spans
from validated segment lengths instead of trusting handwritten spans.

### ADR-Style Decision

**Status:** accepted

#### Context

The Retron Hairpin work spans multiple packages and responsibilities:

- `Cruncher.snapback` handles snapback/released-product topology, search, and
  projection.
- `Cruncher.scar_nick` handles Type IIS retained scar and terminal nick
  feasibility.
- YIU remains a contrast and mismatch language; it must not become a topology
  solver.
- `Construct` currently operates mostly in template-anchor and normalize-anchor
  modes.
- `BaseRender` already follows a contract-first rendering posture and should
  remain a renderer, not a scientific analysis engine.
- USR is a plausible durable sequence store, but the first slice can be local
  artifact bundles while schemas stabilize.

Construct currently persists lineage around templates, anchors, windows,
parts, orientation, and parent-forward IDs. This composition use case needs
component-span lineage instead of anchor/window lineage.

#### Decision

Implement a generic `linear_ssdna_composition` capability in Construct backed
by a neutral shared contract, `linear_ssdna_composition_v1`.

Construct will:

- parse and validate a strict composition contract;
- resolve literal sequence parts first;
- support future source refs to Cruncher artifacts, USR rows, or study records
  without private imports;
- assemble ordered physical segments into a linear ssDNA sequence;
- expand repeated units deterministically;
- emit product-level spans for physical segments, semantic annotations, copy
  boundaries, intended pair maps, provenance, validation results, and export
  artifacts;
- generate the scoped visual contract via `sequence_evidence_map_v1` as a
  canonical component-unit map, not a repeat-expanded map;
- copy any study-owned display profile into the visual contract metadata so
  downstream renderers can style and label artifacts without knowing Retron
  biology;
- optionally persist to USR only after local artifact contracts are stable.

A separate folding layer will:

- consume the assembled sequence artifact;
- run a declared ViennaRNA interface or another backend when configured;
- emit `secondary_structure_prediction_v1`;
- fail explicitly on missing backend, malformed output, length mismatch, or
  parse errors;
- treat folding as advisory by default, hard-gated only when configured.

BaseRender will:

- consume visual contracts;
- render SVG/PDF/PNG artifacts;
- never invoke ViennaRNA interfaces or infer biological topology on its own.

Retron-specific biological selection, ranking, and rationale remain outside
Construct core. The first slice uses explicit manual combinations, not an
opaque "generate every possible hairpin" command.

The Retron study-local compiler will:

- parse lab-facing labels such as
  `pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM`;
- recompute the scar-nick `S3/S2/S1/S0` profile from left/right bases;
- fail fast if a provided profile drifts, if `S0` is not ligatable, or if the
  payload/cap is absent from the study registry;
- join route metadata from
  `docs/studies/retron_hairpin_design/compiler/msd_design_registry.yaml`;
- emit frozen `msd_design_reference_v1` records and batch
  `msd_design_catalog_v1` files into an explicit caller-chosen directory;
- leave artifact paths and sequence digests nullable until a later
  assembly/visual/GenBank slice attaches concrete products.

This layer is study-owned code. Do not add a top-level `retron-msd` script,
and do not make Folding or Construct own persistent Retron MSD workspaces.

#### Consequences

Positive consequences:

- Construct remains generic and reusable for other linear ssDNA products.
- Cruncher does not become a monolithic Retron assembly tool.
- Component identity, spans, repeats, transforms, and provenance are
  first-class.
- Human and agent workflows can author explicit composition specs.
- USR persistence can be added later without forcing USR complexity into the
  tracer bullet.
- Folding and visual QA can be added without making BaseRender a scientific
  computation layer.

Costs and tradeoffs:

- A new contract and Construct mode/API surface are required.
- Schema design must avoid turning `role` strings into hidden Retron
  semantics.
- Folding backend/preflight behavior is a new execution surface.
- Benchling-friendly annotated sequence export needs explicit support.

#### Alternatives Considered And Rejected

| Alternative | Rejection rationale |
| --- | --- |
| Make Cruncher own whole Retron assembly | Overfits Cruncher to one study and turns Snapback/scar_nick primitives into monolithic Retron construction. |
| Add Retron-specific models to Construct | Construct should know segments, annotations, transforms, repeats, topology, and provenance, not TetO/TetR biology. |
| Overload existing template-anchor mode | This creates confusing anchor/window lineage for a non-anchor composition problem. |
| Put ViennaRNA execution inside BaseRender | BaseRender should render visual contracts; folding is scientific analysis. |
| Make ViennaRNA folding always required | Folding is important QA but should be advisory unless configured as a gate. |
| Start with USR overlays as mandatory | Local artifact bundles are lower-risk for the first slice. |
| Add a generic top-level utilities/orchestrator layer | Ownership should stay in packages and shared contracts. |

### Goals And Non-Goals

Goals:

- Add a generic `linear_ssdna_composition` mode or public workflow under
  Construct.
- Define strict, versioned schemas for composition, folding, and visual
  handoff.
- Support literal/manual sequence components as first-class inputs.
- Support deterministic multicopy expansion.
- Preserve physical segment spans and semantic annotations separately.
- Support declared transforms, especially reverse complement.
- Preserve provenance to manual specs, future Cruncher artifacts, USR rows, and
  study records.
- Emit deterministic local artifact bundles.
- Emit Benchling-oriented annotated sequence exports, with GenBank as the first
  target.
- Enable folding QA on the canonical component unit.
- Keep folding advisory by default, hard-gated only when configured.
- Reuse `sequence_evidence_map_v1` for canonical visual QA, with a fail-fast
  boundary against repeat-expanded visual/folding evidence maps.
- Render canonical component-span QA as a paired top/bottom-strand evidence
  view: component color is a coordinate backdrop spanning both sequence rows,
  Watson-Crick correspondence is shown by one light vertical connector per
  position, and semantic annotations remain text labels or data attributes
  rather than duplicate filled boxes.
- Use ViennaRNA-native secondary-structure plotting for fold-layout visuals,
  then layer dnadesign ontology annotations onto those outputs rather than
  rebuilding the structure layout engine in BaseRender.
- Keep package boundaries strict and test-enforced.

Non-goals:

- Do not make Construct Retron-specific.
- Do not make Cruncher own Retron assembly directly.
- Do not make YIU solve topology.
- Do not make BaseRender run ViennaRNA interfaces directly.
- Do not implement a from-scratch secondary-structure layout engine in
  BaseRender for ViennaRNA-predicted folds.
- Do not introduce hidden fallback behavior or implicit path guessing.
- Do not import sibling internal modules such as `dnadesign.cruncher.src.*`
  from Construct or BaseRender.
- Do not require USR persistence in the first implementation slice.
- Do not build an opaque "generate every possible hairpin" command.
- Do not expose the Retron MSD reference compiler as a generic top-level CLI.
- Do not create per-design Construct or Folding workspaces for ID-to-reference
  catalog compilation.
- Do not claim exact current `de033` or scar-nick enzyme-route results without
  rerunning current commands.

### Current-State Summary

Construct is currently a strict sequence realization layer with
`realize_template` and `normalize_anchor` modes. Its lineage is oriented around
templates, anchors, windows, parts, and parent-forward IDs. Add composition as
a new mode rather than forcing linear ssDNA assembly through template-anchor
semantics.

Cruncher is a multi-family workflow system. Retron-relevant surfaces include
`snapback`, `scar_nick`, enzyme catalog/scanning surfaces, YIU, app/CLI
orchestration, and checked-in study records. Cruncher may produce artifacts and
source refs that Construct can import later, but it should not assemble whole
Retron multicopy products directly.

The Retron Hairpin study is the right place for route posture, selected
variants, rationale, dogfooding examples, and current artifact links. Any
Retron-specific selector or ranker should remain study-owned code, not
Construct core.

BaseRender is contract-first and adapter-based. It should consume visual
contracts emitted by Construct/folding adapters and should not run
ViennaRNA itself.

USR is plausible as a central sequence location, but the first slice should
prefer local artifact bundles unless USR adds clear stability without
complexity.

Shared producer/consumer schemas belong in `src/dnadesign/contracts/` when
neutral cross-tool handoff is needed.

### Proposed Architecture

```text
Manual spec / study selector / future Cruncher artifact refs
        |
        v
linear_ssdna_composition_v1
        |
        v
Construct linear ssDNA composer
        |
        +--> assembled_sequence.json
        +--> segment_spans.json
        +--> annotation_spans.json
        +--> provenance.json
        +--> validation_report.json
        +--> sequence.fa
        +--> sequence.gb
        +--> visual/sequence_evidence_map_v1.json
        |
        v
secondary-structure folding layer
        |
        +--> folding/secondary_structure_input_sequence.json
        +--> secondary_structure_prediction_v1.json
        +--> folding_validation_report.json
        |
        v
visual publishers
        |
        +--> component evidence SVG/PDF/PNG
        +--> ViennaRNA-native structure SVG/PDF/PNG plus dnadesign annotation manifest
        |
        v
study record links / optional USR persistence
```

No new code should import sibling private modules such as:

```python
from dnadesign.cruncher.src.snapback import ...
from dnadesign.baserender.src import ...
from dnadesign.construct.src import ...
```

Cross-tool access must happen through shared contracts, public APIs, files,
artifact bundles, USR rows, study records, or explicitly versioned source refs.

### Retron-43 Composition Ontology

The retron-43 literal example is:

```text
gtcagaaaaaaCAAGtccctatcagtgatagagatCCTCAGcccGCTGAGGatctctatcactgatagggaCTCGacagtaactcaga
```

Using the literal string above and zero-based half-open coordinates, the unit
length is 88 nt:

| Span | Physical segment | Sequence |
| ---: | --- | --- |
| `0-15` | `flank_5p` | `gtcagaaaaaaCAAG` |
| `15-34` | `payload_primary` | `tccctatcagtgatagaga` |
| `34-52` | `snapback_foldback_geometry` | `tCCTCAGcccGCTGAGGa` |
| `52-71` | `payload_complement` | `tctctatcactgataggga` |
| `71-88` | `flank_3p` | `CTCGacagtaactcaga` |

Nested semantic annotations include:

| Span | Annotation | Meaning |
| ---: | --- | --- |
| `11-15` | `stem_base_left` | last four bases of the 5-prime flank, `CAAG` |
| `15-34` | `teto_primary` | TetO payload arm |
| `34-41` | `snapback_retained_stem` | retained foldback stem subsection |
| `41-44` | `snapback_cap` | 3 nt cap subsection |
| `44-52` | `snapback_foldback_return` | foldback return subsection |
| `52-71` | `teto_complement` | reverse-complement payload arm |
| `71-75` | `stem_base_right` | first four bases of the 3-prime flank, `CTCG` |

Earlier shorthand spans used a 0-87 layout. The implementation must compute
spans from segment lengths and reject mismatched declared spans.

### Type IIS Scar-Nick Projection Rule

`scar_nick` artifacts model Type IIS scar plus terminal nick feasibility. They
may include recognition sites, nickase footprints, protected/discarded strand
burden, downstream degenerate symbols, and visualization context. Those fields
are feasibility and provenance context, not final linear ssDNA sequence unless
a composition spec explicitly selects a span.

For this Retron composition workflow, only the four-base left and right basal
spans enter the final multicopy ssDNA product:

- `left_base` becomes or annotates the final four bases at the 5-prime flank
  side, for example `CAAG`.
- `right_base` becomes or annotates the first four bases at the 3-prime flank
  side, for example `CTCG`.
- The Type IIS recognition sequence does not enter the final output sequence.
- Any sequence left of the retained four-base span in the scar-nick processing
  model does not enter the output sequence.
- Nickase/release-enzyme burden remains provenance, QA, or visual context.

The four-base spans are the sticky-overhang/base-junction semantics used for
cloning an insert into an expression vector that houses the multicopy ssDNA.
Construct should model them as ordinary sequence spans plus annotations, not as
Type IIS biology.

### `linear_ssdna_composition_v1`

Recommended schema location:

```text
src/dnadesign/contracts/sequence/linear_ssdna_composition_v1.py
```

or, if the repo prefers tool-namespaced contracts:

```text
src/dnadesign/contracts/construct/linear_ssdna_composition_v1.py
```

Illustrative input YAML:

```yaml
contract: linear_ssdna_composition_v1
schema_version: 1

composition_id: retron43_teto_manual_x8
alphabet: dna
topology: linear_ssdna
coordinate_system: zero_based_half_open
case_policy: preserve_input_display_case
canonicalization:
  compare_sequences_case_insensitive: true
  output_sequence_preserves_case: true

units:
  - unit_id: retron43_teto_unit
    repeat_count: 8
    segments:
      - segment_id: flank_5p
        role: flank_5p
        sequence: gtcagaaaaaaCAAG
        source:
          kind: literal
          label: manual_retron43_example

      - segment_id: payload_primary
        role: payload_primary
        sequence: tccctatcagtgatagaga
        source:
          kind: literal
          label: manual_teto_payload

      - segment_id: snapback_foldback_geometry
        role: snapback_foldback_geometry
        sequence: tCCTCAGcccGCTGAGGa
        source:
          kind: literal
          label: manual_snapback_43_cap

      - segment_id: payload_complement
        role: payload_complement
        sequence: tctctatcactgataggga
        transform:
          kind: reverse_complement
          source_segment_id: payload_primary
          assert_expected_sequence: true
        source:
          kind: derived
          from_segment_id: payload_primary

      - segment_id: flank_3p
        role: flank_3p
        sequence: CTCGacagtaactcaga
        source:
          kind: literal
          label: manual_retron43_example

    annotations:
      - annotation_id: stem_base_left
        role: stem_base_left
        location:
          basis: segment
          segment_id: flank_5p
          start: 11
          end: 15

      - annotation_id: teto_primary
        role: payload
        semantic_label: TetO
        location:
          basis: segment
          segment_id: payload_primary
          start: 0
          end: 19

      - annotation_id: snapback_cap
        role: snapback_cap
        location:
          basis: segment
          segment_id: snapback_foldback_geometry
          start: 0
          end: 18

      - annotation_id: teto_complement
        role: payload_complement
        semantic_label: TetO_reverse_complement
        location:
          basis: segment
          segment_id: payload_complement
          start: 0
          end: 19

      - annotation_id: stem_base_right
        role: stem_base_right
        location:
          basis: segment
          segment_id: flank_3p
          start: 0
          end: 4

    assertions:
      - assertion_id: payload_rc
        kind: reverse_complement
        left_segment_id: payload_primary
        right_segment_id: payload_complement
        severity: error

qa:
  require_no_unknown_bases: true
  allow_degenerate_bases: false
  require_segment_span_coverage: true
  require_non_overlapping_physical_segments: true
  require_annotation_bounds: true
  require_declared_transform_checks: true
  allow_cross_copy_intended_pairings: false

folding:
  enabled: true
  required: false
  scope: canonical_component_unit
  backend:
    name: ViennaRNA
    interface: python_api
    python_module: RNA
    backend_contract: secondary_structure_prediction_v1
    parameters:
      temperature_c: 37.0
  dna_policy:
    mode: convert_t_to_u_for_rna_backend

visual:
  emit:
    - sequence_evidence_map_v1
    - viennarna_secondary_structure_svg_v1
  viennarna_structure_plot:
    layout_algorithm: naview
    emphasize_stem_base_nucleotides: true
  render_exports:
    formats: [svg, pdf, png]

benchling_export:
  enabled: true
  primary_format: genbank
  sidecars: [fasta, features_csv]

output:
  workspace: workspaces/construct/retron43_teto_manual_x8
  artifact_bundle: artifacts/construct/retron43_teto_manual_x8
  usr:
    enabled: false
```

Key schema rules:

- `contract` and `schema_version` must be supported exactly.
- `alphabet` initially supports `dna`.
- `topology` initially supports `linear_ssdna`.
- `coordinate_system` is explicit; use `zero_based_half_open`.
- `segments` assemble the physical sequence in listed order.
- `annotations` may overlap and may be segment-relative or unit-relative.
- `role` and `semantic_label` are metadata labels; Construct validates shape,
  not Retron meaning.
- Source refs are provenance references, not private imports.
- Transforms may derive sequence or assert a provided sequence.
- Repeats expand deterministically and emit copy-level spans.

Illustrative output JSON:

```json
{
  "contract": "linear_ssdna_composition_v1",
  "schema_version": 1,
  "composition_id": "retron43_teto_manual_x8",
  "status": "ok",
  "alphabet": "dna",
  "topology": "linear_ssdna",
  "coordinate_system": "zero_based_half_open",
  "sequence": {
    "id": "retron43_teto_manual_x8",
    "length": 704,
    "sha256": "<digest>",
    "sequence": "<8 copies concatenated>"
  },
  "unit_copies": [
    {
      "unit_id": "retron43_teto_unit",
      "copy_index": 0,
      "span": {"start": 0, "end": 88}
    }
  ],
  "segment_spans": [
    {
      "copy_index": 0,
      "segment_id": "flank_5p",
      "role": "flank_5p",
      "span": {"start": 0, "end": 15},
      "sequence": "gtcagaaaaaaCAAG"
    }
  ],
  "annotation_spans": [
    {
      "copy_index": 0,
      "annotation_id": "stem_base_left",
      "role": "stem_base_left",
      "span": {"start": 11, "end": 15},
      "sequence": "CAAG"
    }
  ],
  "assertions": [
    {
      "assertion_id": "payload_rc",
      "kind": "reverse_complement",
      "status": "pass"
    }
  ],
  "provenance": {
    "inputs": [],
    "source_refs": [],
    "created_by": "construct.linear_ssdna_composition"
  },
  "artifacts": {
    "fasta": "sequence.fa",
    "genbank": "sequence.gb",
    "features_csv": "features.csv",
    "validation_report": "validation_report.json",
    "visual_contract": "visual/sequence_evidence_map_v1.json"
  }
}
```

Terminology note: earlier context sometimes used `component_spans`. In this
spec, physical assembled components are persisted as `segment_spans`; semantic
component meanings are persisted as `annotation_spans`. If an implementation
uses `component_spans` as a compatibility alias, it must point at the same
physical segment-span contract and not create a third ontology.

### `secondary_structure_prediction_v1`

Recommended schema location:

```text
src/dnadesign/contracts/folding/secondary_structure_prediction_v1.py
```

The folding result must reference the exact assembled sequence digest and
record backend package name, interface, executable or module entrypoint,
version, command, parameters, DNA/RNA policy, dot-bracket string, parsed pair
map, energy, stdout/stderr refs, warnings, and errors.

Rules:

- Dot-bracket length must equal folded sequence length.
- Pair map coordinates must map back to original assembled DNA coordinates.
- Missing backend is an explicit preflight state.
- Malformed output is an explicit error.
- If `required: false`, assembly remains valid but folding result is `error`
  or `not_run`.
- If `required: true`, folding errors fail the configured pipeline step.

The default DNA/RNA policy must be reject-without-policy. The request must
declare whether the backend accepts DNA directly or whether `T` is converted to
`U` for an RNA backend and then mapped back to DNA coordinates.

Illustrative folding request:

```yaml
contract: secondary_structure_prediction_request_v1
schema_version: 1

input:
  sequence_artifact: secondary_structure_input_sequence.json
  sequence_id: retron43_teto_manual_x8.component_span_qa
  sequence_sha256: "<digest>"
  alphabet: dna
  topology: linear_ssdna

scope:
  mode: canonical_component_unit

backend:
  name: ViennaRNA
  interface: python_api
  python_module: RNA
  parameters:
    temperature_c: 37.0
  dna_policy:
    mode: convert_t_to_u_for_rna_backend
    output_coordinates: original_dna_sequence

policy:
  required: false
  fail_on_malformed_output: true
  fail_on_length_mismatch: true
```

Illustrative folding output:

```json
{
  "contract": "secondary_structure_prediction_v1",
  "schema_version": 1,
  "prediction_id": "retron43_teto_manual_x8.rnafold.full",
  "status": "ok",
  "input": {
    "sequence_id": "retron43_teto_manual_x8",
    "sequence_sha256": "<digest>",
    "alphabet": "dna",
    "topology": "linear_ssdna",
    "length": 704
  },
  "backend": {
    "name": "ViennaRNA",
    "version": "<captured version>",
    "command": ["RNA.fold_compound", "mfe"],
    "parameters": {
      "temperature_c": 37.0
    }
  },
  "dna_policy": {
    "mode": "convert_t_to_u_for_rna_backend",
    "submitted_alphabet": "rna_surrogate",
    "coordinates_mapped_to": "original_dna_sequence"
  },
  "result": {
    "dot_bracket": "...",
    "mfe_kcal_mol": -12.3,
    "pair_map": [
      {"left": 12, "right": 74, "pair": "AU"}
    ]
  },
  "qa": {
    "length_matches_input": true,
    "cross_copy_pairings": [],
    "warnings": []
  },
  "artifacts": {
    "stdout": "folding/ViennaRNA.python_api.stdout.txt",
    "stderr": "folding/ViennaRNA.python_api.stderr.txt"
  }
}
```

Equivalent CLI requests may select `interface: cli` with
`executable: RNAfold`; in that case `RNAfold` is the ViennaRNA command-line
program and the backend package name remains `ViennaRNA`.

### Visual Contract Strategy

Start with `sequence_evidence_map_v1`.

Use it for:

- canonical component-unit sequence evidence;
- physical segment spans;
- semantic annotation spans;
- intended pairings when declared;
- predicted pairings from `secondary_structure_prediction_v1`;
- source/provenance labels.

For secondary-structure layout, prefer ViennaRNA-native plotting. ViennaRNA is
already the package that computes the fold and provides structure plotting
surfaces: the `RNAplot` CLI can draw structure graphs in EPS, SVG, GML, and
XRNA formats, and the Python API exposes `RNA.svg_rna_plot`,
`RNA.plot_structure_svg`, and layout helpers. The first dnadesign fold visual
publisher should therefore:

- read `secondary_structure_prediction_v1`, the canonical folding sequence,
  and `sequence_evidence_map_v1`;
- render the native ViennaRNA SVG through the uv-managed `RNA` Python module;
- validate that the SVG contains one addressable nucleotide node per input
  coordinate and addressable base-pair nodes for predicted pairs;
- post-process the SVG DOM to attach dnadesign zero-based coordinates,
  one-based ViennaRNA display coordinates, owner/effect classes, and
  component hues;
- optionally emphasize annotated left/right stem-base nucleotides with a
  bold/stroked text style, defaulting on for Snapback/scar-nick QA and
  opt-out through `visual.viennarna_structure_plot`;
- keep section-label and subtitle typography on one pinned annotation size so
  the structure summary does not read as a separate, smaller caption system;
- emit a `viennarna_secondary_structure_svg_v1` manifest that records native
  SVG, annotated SVG, annotation manifest, layout algorithm, backend version,
  source prediction, and source visual contract.

BaseRender remains the renderer for linear component-span QA from visual
contracts. It should not become a duplicate secondary-structure layout engine.
Only add a non-ViennaRNA layout contract, such as `secondary_structure_visual_v1`,
if a future representation is intentionally orthogonal to the native
ViennaRNA structure graph.

BaseRender inputs are visual contracts. BaseRender outputs are SVG, PDF, and
PNG. BaseRender must not call ViennaRNA interfaces, resolve Cruncher candidate
artifacts, assemble sequences, or infer Retron biology.

### Optional USR Strategy

First slice:

- Write local artifact bundles only.
- Do not require USR.
- Include stable IDs, sequence digest, and source refs so artifacts can later
  be imported.

Future USR slice:

- Store final assembled sequence as a USR row.
- Add composition metadata columns or overlays such as
  `construct__composition_id`, `construct__composition_contract`,
  `construct__topology`, `construct__copy_count`,
  `construct__segment_spans_ref`, `construct__annotation_spans_ref`,
  `construct__folding_ref`, and `construct__visual_contract_ref`.
- Keep overlays record-scoped and cheap.
- Use preflight for USR root availability, permissions, and conflict policy.

### Config And CLI UX

The first ergonomic surface should be config-first:

```bash
dnadesign construct compose validate configs/retron43_teto_manual_x8.yaml
dnadesign construct compose run configs/retron43_teto_manual_x8.yaml
```

Equivalent command names may be adjusted to match the existing Construct CLI,
but the behavior should preserve `validate`, `preflight`, `run`, and `export`
surfaces.

Avoid opaque brute-force commands such as:

```bash
dnadesign construct generate-all-retron-hairpins
```

Future source refs should be declarative:

```yaml
source:
  kind: cruncher_artifact
  contract: scar_nick_candidate_v1
  uri: ../../cruncher/workspaces/scar_nick_teto/artifacts/candidates.json
  selector:
    candidate_id: ...
  resolution:
    mode: declared_only
  projection:
    include_final_sequence_spans_only: true
    selected_fields: [left_base, right_base]
```

Resolution modes:

| Mode | Meaning |
| --- | --- |
| `declared_only` | Keep as provenance; do not read file. |
| `required` | Resolve artifact and fail if unavailable. |
| `optional` | Resolve if available; emit warning if missing. |

Construct must never import Cruncher private Python models to resolve these.

For scar-nick source refs, the first supported projection is intentionally
narrow:

```yaml
source:
  kind: cruncher_artifact
  contract: scar_nick_candidate_v1
  uri: ../../cruncher/workspaces/scar_nick_teto/artifacts/candidates.json
  selector:
    candidate_id: ...
  resolution:
    mode: declared_only
  projection:
    include_final_sequence_spans_only: true
    selected_fields: [left_base, right_base]
    excluded_from_final_sequence:
      - type_iis_recognition_sequence
      - nickase_footprint
      - release_enzyme_footprint
      - upstream_processing_context_left_of_retained_scar
      - downstream_degenerate_symbols
```

That projection prevents a future implementation from accidentally
concatenating the Type IIS route context into the final ssDNA insert.

### Artifact Layout

Recommended local bundle:

```text
workspaces/construct/retron43_teto_manual_x8/
  configs/
    composition.yaml
  artifacts/
    manifest.json
    assembled_sequence.json
    sequence.fa
    sequence.gb
    features.csv
    segment_spans.json
    annotation_spans.json
    provenance.json
    validation_report.json
    visual/
      sequence_evidence_map_v1.json
      component_map.svg
      component_map.pdf
      component_map.png
    folding/
      secondary_structure_input_sequence.json
      request.yaml
      secondary_structure_prediction_v1.json
      ViennaRNA.python_api.stdout.txt
      ViennaRNA.python_api.stderr.txt
      predicted_structure.svg
      predicted_structure.pdf
      predicted_structure.png
  reports/
    summary.md
```

Required first-slice artifacts:

| Artifact | Purpose |
| --- | --- |
| `manifest.json` | Bundle index, contract versions, paths, status. |
| `assembled_sequence.json` | Final sequence, length, digest, topology, alphabet, config digest. |
| `sequence.fa` | FASTA handoff. |
| `segment_spans.json` | Physical, non-overlapping assembled segments per copy. |
| `annotation_spans.json` | Semantic annotations, allowed to overlap. |
| `provenance.json` | Literal inputs, source refs, transform derivations, config digest. |
| `validation_report.json` | Pass/fail/warnings with explicit error codes. |
| `visual/sequence_evidence_map_v1.json` | Canonical component-span QA contract for BaseRender and ViennaRNA annotation. It renders one representative unit, not every repeat copy. Exact-span annotations that duplicate physical segments are recorded as suppressed metadata instead of rendered twice. |
| `folding/secondary_structure_input_sequence.json` | Canonical component-unit sequence used by folding requests. |
| `sequence.gb` | Benchling-oriented GenBank export. |
| `features.csv` | Feature table sidecar. |

Example `segment_spans.json`:

```json
{
  "contract": "linear_ssdna_segment_spans_v1",
  "coordinate_system": "zero_based_half_open",
  "sequence_id": "retron43_teto_manual_x1",
  "sequence_length": 88,
  "segments": [
    {"copy_index": 0, "segment_id": "flank_5p", "start": 0, "end": 15},
    {"copy_index": 0, "segment_id": "payload_primary", "start": 15, "end": 34},
    {"copy_index": 0, "segment_id": "snapback_foldback_geometry", "start": 34, "end": 52},
    {"copy_index": 0, "segment_id": "payload_complement", "start": 52, "end": 71},
    {"copy_index": 0, "segment_id": "flank_3p", "start": 71, "end": 88}
  ]
}
```

Example `annotation_spans.json`:

```json
{
  "contract": "linear_ssdna_annotation_spans_v1",
  "coordinate_system": "zero_based_half_open",
  "sequence_id": "retron43_teto_manual_x1",
  "annotations": [
    {"copy_index": 0, "annotation_id": "stem_base_left", "role": "stem_base_left", "start": 11, "end": 15},
    {"copy_index": 0, "annotation_id": "teto_primary", "role": "payload", "semantic_label": "TetO", "start": 15, "end": 34},
    {"copy_index": 0, "annotation_id": "snapback_cap", "role": "snapback_cap", "start": 34, "end": 52},
    {"copy_index": 0, "annotation_id": "teto_complement", "role": "payload_complement", "start": 52, "end": 71},
    {"copy_index": 0, "annotation_id": "stem_base_right", "role": "stem_base_right", "start": 71, "end": 75}
  ]
}
```

GenBank should be the first annotated export target because it can carry named
features and is broadly compatible with sequence tools. Convert internal
zero-based half-open spans to one-based inclusive GenBank locations:

| Internal span | GenBank location | Feature label |
| --- | ---: | --- |
| `0-15` | `1..15` | `flank_5p` |
| `11-15` | `12..15` | `stem_base_left` |
| `15-34` | `16..34` | `payload_primary / TetO` |
| `34-52` | `35..52` | `snapback_foldback_geometry` |
| `52-71` | `53..71` | `payload_complement` |
| `71-88` | `72..88` | `flank_3p` |
| `71-75` | `72..75` | `stem_base_right` |

Sidecars:

- FASTA for simple sequence copy/paste.
- CSV feature table for spreadsheet inspection and Benchling fallback import.
- JSON contracts as source of truth.

### Validation And Preflight Rules

Schema validation fails fast for unknown top-level keys, unsupported contracts
or schema versions, unsupported topology, missing IDs, duplicate IDs, missing
segment refs, and malformed transforms/assertions.

Sequence validation:

- Accept `A/C/G/T` case-insensitively.
- Preserve input case for display/export.
- Compare canonical uppercase for validation.
- Reject `N` and IUPAC degenerates unless `allow_degenerate_bases: true`.
- Reject RNA `U` in `alphabet: dna` unless an explicit policy allows
  conversion.
- Reject circular/duplex topology for this contract version.

Physical segments:

- assemble in listed order;
- are contiguous and non-overlapping within each copy;
- cover the full unit copy span;
- may not have zero length.

Annotations:

- may overlap physical segments or other annotations;
- must remain within referenced segment, unit, or product bounds;
- may not have zero length.

Reverse-complement checks:

- are enforced only when declared;
- compare canonical uppercase DNA;
- derive sequence deterministically if only `transform` is supplied;
- fail if both `sequence` and `transform` are supplied but disagree.
- do not infer that stem bases are reverse complements unless the spec
  declares it.

For the retron-43 example, `payload_complement` should pass:

```text
payload_primary:    tccctatcagtgatagaga
reverse complement: tctctatcactgataggga
```

Copy expansion:

- `repeat_count` must be a positive integer;
- final length must equal the sum of expanded unit lengths;
- every segment and annotation span maps to a copy unless explicitly
  product-level.

Folding preflight states:

| State | Meaning |
| --- | --- |
| `ok` | Backend available, version captured, output directory writable. |
| `warning_optional_missing` | Backend missing but folding is advisory. |
| `blocker_required_missing` | Backend missing and folding is required. |
| `blocker_policy_unknown` | DNA/RNA backend policy is absent or unsupported. |
| `blocker_output_unwritable` | Folding output path cannot be written. |

Backend interfaces:

- prefer `backend.interface: python_api` with `python_module: RNA` from the
  uv-managed `viennarna` package for reproducible local dogfooding;
- keep `backend.interface: cli` for workflows that explicitly need a
  system-provided ViennaRNA `RNAfold` executable;
- record the submitted sequence alphabet and coordinate mapping for both
  interfaces.

Malformed folding output behavior:

- nonzero exit code: error;
- missing dot-bracket: error;
- dot-bracket length mismatch: error;
- unparseable MFE: warning or error based on policy;
- invalid bracket nesting: error;
- pair map out of bounds: error;
- empty output: error.

Do not silently fail open. If folding is advisory, the assembly artifact may
remain `ok`, but folding status must be explicit.

### Folding QA Design

The folding layer should be a backend runner plus parser, not Construct core
and not BaseRender. Candidate package names are `src/dnadesign/folding/` or
`src/dnadesign/secondary_structure/`; choose the exact path during
implementation after checking existing package conventions.

ViennaRNA can be the first backend, but the contract should stay
backend-neutral. The default implementation path uses the official Python API
via `RNA.fold_compound(...).mfe()` under uv; a system-provided ViennaRNA
`RNAfold` executable remains an optional CLI interface. Folding must run on the
canonical component unit for this workflow; repeat-expanded folding is deferred
until there is an explicit, separate analysis contract.

Default policy:

```yaml
folding:
  enabled: true
  required: false
```

If the selected ViennaRNA interface is missing and folding is advisory,
assembly can still succeed, but the folding artifact must be explicit
`warning_optional_missing`, `not_run`, or `error` rather than silently green.

Report at minimum:

- dot-bracket length matches canonical component-unit sequence length;
- MFE captured;
- pair map parsed and in bounds;
- intended declared pairings recovered or missed;
- cross-copy pairings count, expected to be zero for the canonical unit unless
  a future multi-record analysis contract says otherwise;
- pairings involving annotated payload/stem/cap regions;
- backend warnings/errors.

`visual.emit` controls whether Construct publishes the optional native
ViennaRNA structure plot. Folding still emits and enriches
`secondary_structure_prediction_v1` when folding succeeds, but
`viennarna_secondary_structure_svg_v1` artifacts are emitted only when that
contract kind is explicitly present in `visual.emit`. Layout selection is
recorded through `visual.viennarna_structure_plot.layout_algorithm`; stem-base
nucleotide emphasis is controlled by
`visual.viennarna_structure_plot.emphasize_stem_base_nucleotides`.

### Visual QA Design

The first visual artifact should show the canonical 5-prime-flank through
3-prime-flank component unit with physical segment blocks, semantic annotation
overlays, payload/complement relationship, source labels, and validation
markers. It should not concatenate repeat copies for visual QA.

When Construct publishes a compound review that embeds the standalone
BaseRender component-span SVG below the ViennaRNA plot, it should omit the
standalone component-span title in that embedded lower row. The source
BaseRender artifact may keep its own title for standalone inspection, but the
compound overview should use the upper structure title/subtitle as the single
composition identifier.

The predicted-structure visual should be driven by
`secondary_structure_prediction_v1` and should use ViennaRNA-native secondary
structure plotting as the layout source of truth. The dnadesign layer may add
annotation-colored nucleotides, intended versus predicted pair distinctions,
component metadata and intended-pair categories, but it must not
replace ViennaRNA's structure graph layout with a hand-rolled BaseRender layout.

Implementation note: native ViennaRNA SVG internals are useful but not a
stable dnadesign contract. Tests must pin the observed `viennarna==2.7.2`
surface by asserting nucleotide node count, sequence-order coordinate mapping,
base-pair ID parsing, and annotated SVG validity.

Render export priority:

1. SVG as source-of-truth visual output.
2. PDF for presentation and QA records.
3. PNG for quick sharing and reports.

### Study Integration

The Retron Hairpin study should record selected payloads, flanks,
snapback/cap segments, scar-nick stem-base candidates, rationale, source
artifact refs, composition config path, Construct output bundle path, folding
result path, visual output path, Benchling export path, known risk notes, and
freshness of command outputs used for claims.

Start selector/ranking as study config and rationale. Promote it to code only
when repeated behavior exists, such as selecting compatible Snapback and
scar-nick candidates, ranking payload/cap/stem-base combinations, or producing
composition specs from public Cruncher artifacts.

Dogfooding layers:

1. Manual retron-43/TetO composition using literal sequences.
2. Optional `de033` plus `scar_nick_teto` demo using source refs.

For the second, do not claim exact current route hits unless fresh command
output supports the claim.

### Testing Plan

Schema tests:

- accept minimal valid literal composition;
- reject unknown keys;
- reject unsupported contract/schema/topology;
- reject duplicate segment IDs;
- reject missing transform source segment;
- reject zero-length segment;
- reject annotation outside segment bounds;
- reject invalid alphabet characters by default;
- accept mixed case while canonicalizing comparisons.

Folding contract tests:

- accept valid backend result;
- reject dot-bracket length mismatch;
- reject invalid pair map coordinates;
- reject malformed bracket strings;
- require backend name/version/parameters;
- require explicit DNA/RNA policy.

Contract round-trip tests:

```text
YAML/JSON fixture -> model parse -> canonical JSON -> parse again -> equality/digest stable
```

Retron-43 golden assertions:

- assembled sequence equals expected concatenation;
- payload complement equals reverse complement of payload primary;
- segment spans are contiguous and non-overlapping;
- annotation spans are valid and may overlap segments;
- copy count expansion is deterministic;
- GenBank locations are converted correctly;
- visual contract includes expected layers;
- only four-base `scar_nick` left/right spans are projected into final ssDNA
  from scar-nick source refs.

Minimal synthetic fixture:

```yaml
segments:
  - segment_id: left
    sequence: AAAA
  - segment_id: payload
    sequence: ACGT
  - segment_id: payload_rc
    sequence: ACGT
    transform:
      kind: reverse_complement
      source_segment_id: payload
  - segment_id: right
    sequence: TTTT
repeat_count: 3
```

Architecture-boundary tests should fail if Construct imports
`dnadesign.cruncher.src.*`, BaseRender imports Construct or Cruncher internals,
the folding layer imports BaseRender internals, or a study selector imports
private sibling modules instead of contracts/public APIs.

### Implementation Plan Summary

The completed implementation checklist is in:
[Generic linear ssDNA composition](../../../../exec-plans/completed/2026-05-13-generic-linear-ssdna-composition.md).
Remaining hardening work is tracked in:
[Linear ssDNA composition hardening follow-ups](../../../../exec-plans/active/2026-05-14-linear-ssdna-composition-hardening-followups.md).

High-level phases:

1. ADR and contract skeleton.
2. Construct local composition tracer bullet.
3. Benchling/GenBank and sidecar exports.
4. Visual contract publisher.
5. Folding backend contract and ViennaRNA runner.
6. ViennaRNA-native structure SVG publisher with dnadesign annotation.
7. Optional USR persistence.
8. Study dogfooding and source refs.

### Risks And Open Questions

Risks:

- Current Retron study wording may conflict around exact hits versus bounded
  near-hit posture. Require fresh command output before making exact-hit
  claims.
- Broader scar-nick enzyme-route claims, especially PaqCI, should be rerun
  before being treated as current.
- A system-provided ViennaRNA `RNAfold` executable may not be installed even
  when the uv-managed ViennaRNA Python API is available; requests must declare
  which interface they require.
- DNA/RNA folding policy must be explicit.
- USR may be the right durable sequence location later, but local artifacts are
  the first slice.
- Retron-specific selector logic should start as study config/rationale.
- The retron-43 literal and earlier shorthand spans differ by one nucleotide;
  tests must lock the chosen literal and coordinates.

Open questions:

| Question | Recommendation |
| --- | --- |
| Exact contract namespace under `src/dnadesign/contracts/` | Use `contracts/sequence` if available; otherwise `contracts/construct`. |
| Exact folding package name | Use a separate `folding` or `secondary_structure` package. |
| Exact study-code path for future selectors | Keep as study config first; add a separate ADR if code is needed. |
| Whether to decompose `snapback_foldback_geometry` immediately | Start as one physical segment with explicit cap/stem/return sub-annotations. |
| Whether to assert stem-base complementarity | Only assert if declared. |
| Whether to support heterogeneous multicopy units in v1 | Keep v1 capable through multiple units, but first tracer uses homogeneous `repeat_count`. |

### Links

- Study handoff: [linear ssDNA composition](../../../../studies/retron_hairpin_design/contexts/linear-ssdna-composition.md)
- Study routes: [Retron Hairpin routes](../../../../studies/retron_hairpin_design/routes/README.md)
- Scar-nick context: [scar-nick base-junction](../../../../studies/retron_hairpin_design/contexts/scar-nick-base-junction.md)
- Implementation record: [generic linear ssDNA composition](../../../../exec-plans/completed/2026-05-13-generic-linear-ssdna-composition.md)
- Follow-up plan: [linear ssDNA composition hardening follow-ups](../../../../exec-plans/active/2026-05-14-linear-ssdna-composition-hardening-followups.md)
- Planning lifecycle: [PLANS](../../../../../PLANS.md)
