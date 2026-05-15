## Retron Linear ssDNA Composition Handoff

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-14
**Status:** Construct tracer bullet, BaseRender component-span QA, uv-managed ViennaRNA folding, and ViennaRNA-native annotated structure SVG implemented
**Generic authority:** [Construct linear ssDNA composition reference](../../../src/dnadesign/construct/docs/reference/linear-ssdna-composition.md)
**Dev spec:** [generic linear ssDNA composition](../../dev/plans/2026-05-13-generic-linear-ssdna-composition-spec.md)
**Implementation record:** [completed checklist](../../exec-plans/completed/2026-05-13-generic-linear-ssdna-composition.md)

Retron Hairpin study questions shift here when the work moves from solving
Snapback or scar-nick primitives to composing a whole multicopy linear ssDNA
insert. This page is Retron-specific; use the Construct reference above for
generic composition and bundle-routing procedure.

The current product profile is an ID-to-reference compiler plus single-unit
sequence artifacts. The user provides or selects primitive parts through a
lab-facing MSD shorthand label; study code validates those parts against
registry metadata and scar-nick profile rules; Construct/BaseRender are invoked
as stateless service surfaces. Do not create one persistent Construct or Folding
workspace per requested design.

### Study Boundary

The study owns Retron-specific intent, selected variants, rationale, and
dogfooding links. It does not own the generic assembler.

| Question | Owner |
| --- | --- |
| Which payload, stem-base pair, cap, flank, and output artifact route are selected? | Retron study record |
| How is a lab-facing MSD shorthand label normalized and linted? | Retron study `msd_design_registry.yaml` plus `dnadesign.studies.retron_hairpin_design.cli` |
| How are ordered ssDNA segments concatenated and spans emitted? | Construct |
| Which snapback/cap candidates exist? | Cruncher Snapback |
| Which four-base stem bases are Type IIS scar plus terminal nick feasible? | Cruncher scar_nick |
| How is the component unit folded for QA? | Folding/backend layer |
| How are component spans rendered? | BaseRender from canonical `visual/sequence_evidence_map_v1.json` |
| How are predicted secondary structures rendered? | ViennaRNA-native SVG plus dnadesign annotation manifest |

YIU remains contrast-only. It is not the topology engine and should not be
used to solve this composition problem.

### Composition Ontology

Use this rule:

```text
Segments assemble the sequence; annotations interpret spans.
```

For the study-owned MSD compiler, use the adjacent rule:

```text
IDs select and validate provided parts; catalogs freeze references.
```

The compact label is not the source of truth by itself. The compiler parses it,
recomputes the scar-nick `S3/S2/S1/S0` profile, fails fast on drift, joins
study registry metadata, and writes `msd_design_reference_v1` /
`msd_design_catalog_v1` records. The materialize route then emits one MSD unit
per design and attaches sequence digests plus artifact paths to those records.

For the manual retron-43/TetO dogfood unit, the physical segments are:

| Span | Segment | Sequence |
| ---: | --- | --- |
| `0-15` | `flank_5p` | `gtcagaaaaaaCAAG` |
| `15-34` | `payload_primary` | `tccctatcagtgatagaga` |
| `34-52` | `snapback_cap_segment` | `tCCTCAGcccGCTGAGGa` |
| `52-71` | `payload_complement` | `tctctatcactgataggga` |
| `71-88` | `flank_3p` | `CTCGacagtaactcaga` |

Nested annotations include:

- `stem_base_left`: `11-15`, `CAAG`
- `teto_primary`: `15-34`
- `snapback_cap`: `34-52`
- `teto_complement`: `52-71`
- `stem_base_right`: `71-75`, `CTCG`

The literal unit is 88 nt under zero-based half-open coordinates. Do not reuse
older shorthand spans without recomputing them from the actual segment lengths.

### Scar-Nick Projection Rule

`scar_nick` answers Type IIS scar plus terminal nick feasibility. Its source
artifacts may contain Type IIS recognition sequence, nickase footprint,
downstream degenerate sequence, protected/discarded strand burden, and visual
context. Those are not final ssDNA sequence by default.

For Retron multicopy ssDNA composition:

- only the four-base `left_base` and `right_base` basal spans enter the final
  assembled ssDNA product;
- `left_base` is the left/stem-base annotation at the 5-prime flank side;
- `right_base` is the right/stem-base annotation at the 3-prime flank side;
- Type IIS recognition sequence does not enter the output sequence;
- sequence to the left of the retained four-base span in the scar-nick
  processing model does not enter the output sequence;
- nickase/release-enzyme burden remains provenance or QA context.

The four-base spans are the sticky-overhang/base-junction semantics used for
cloning an insert into an expression vector that houses the multicopy ssDNA.
Construct should treat them as ordinary sequence spans plus annotations.

### First Dogfood Target

Start with a manual literal composition rather than resolving Cruncher outputs:

```text
composition_id: retron43_teto_manual_x8
unit: retron43_teto_unit
repeat_count: 8
topology: linear_ssdna
folding.scope: canonical_component_unit
usr.enabled: false
```

Required outputs:

- assembled sequence JSON and FASTA;
- segment and annotation span JSON;
- validation report;
- GenBank and feature CSV for Benchling handoff;
- canonical component-span QA `sequence_evidence_map_v1` at
  `visual/sequence_evidence_map_v1.json` for BaseRender and ViennaRNA
  annotation. This renders the representative 5-prime-flank-through-3-prime-
  flank unit, not all repeated copies, and suppresses exact-span annotation
  duplicates such as payload labels that already exist as physical segment
  owners. BaseRender renders this as a paired top/bottom-strand view with
  component backdrops spanning both rows, solid light Watson-Crick connectors,
  text-only annotation labels rather than separate annotation rectangles, and
  embossed left/right stem-base glyphs on both strands. The current dogfood
  render carries publication-facing labels (`TetO primary`, `Snapback cap`,
  `TetO complement`, flank and stem-base labels), component hues, and backdrop
  styles in `visual.display_profile` on the study config; Construct and
  Folding consume that profile as data instead of hard-coding Retron terms.
  The render uses one pinned small display font size, stronger component
  backdrops, and collision-safe label tiering close to the relevant strand;
- canonical folding input at
  `folding/secondary_structure_input_sequence.json`;
- generated BaseRender component-span QA job and SVG render path;
- `secondary_structure_prediction_v1` folding request, preflight, and
  prediction artifacts through the uv-managed ViennaRNA Python API. If a future
  config selects a missing advisory CLI backend, the prediction status must be
  explicit rather than silently green.
- ViennaRNA-native secondary-structure SVG artifacts annotated with dnadesign
  component hues, section labels, stem-base highlights, upright
  counter-rotated nucleotide text, coordinate metadata, cap-right orientation
  metadata, label collision metadata, a single fitted dnadesign background
  canvas, closer left/right stem-base labels, optional bold/stroked stem-base
  nucleotide emphasis, and a centered title/subtitle stack when a snapback-cap
  section is present. Section labels and subtitle lines use one pinned
  annotation font size. Visible summary wording should stay concise and
  publication-facing, not machine-key-heavy.
- generated two-row composition review SVG under `visual/reviews/` that stacks
  the ViennaRNA annotated structure above the BaseRender component-span SVG and
  balances visual weight by keeping the folded plot readable while scaling and
  emboldening the component-span row enough that neither subplot visually
  dominates. The review omits the standalone BaseRender component-span title
  because the top title/subtitle already identifies the composition. BaseRender
  remains the linear component-span renderer and should not duplicate
  ViennaRNA's fold layout.

Current checked-in Construct surface:

- config:
  `src/dnadesign/construct/workspaces/retron43_teto_manual_x8/config.composition.yaml`
- validate:
  `uv run construct compose validate --config src/dnadesign/construct/workspaces/retron43_teto_manual_x8/config.composition.yaml`
- run:
  `uv run construct compose run --config src/dnadesign/construct/workspaces/retron43_teto_manual_x8/config.composition.yaml`
- BaseRender component-span QA:
  `uv run baserender job run src/dnadesign/construct/workspaces/retron43_teto_manual_x8/outputs/retron43_teto_manual_x8/baserender_jobs/component_span_qa_svg.yaml`
- composition review SVG:
  `uv run construct compose review --bundle src/dnadesign/construct/workspaces/retron43_teto_manual_x8/outputs/retron43_teto_manual_x8`
- Folding preflight:
  `uv run folding preflight --bundle src/dnadesign/construct/workspaces/retron43_teto_manual_x8/outputs/retron43_teto_manual_x8`
- Folding run:
  `uv run folding run --bundle src/dnadesign/construct/workspaces/retron43_teto_manual_x8/outputs/retron43_teto_manual_x8`
- Folding plot publish:
  `uv run folding plot --bundle src/dnadesign/construct/workspaces/retron43_teto_manual_x8/outputs/retron43_teto_manual_x8 --layout naview --emphasize-stem-bases`
  Bundle mode resolves `folding/secondary_structure_prediction_v1.json`,
  `folding/secondary_structure_input_sequence.json`,
  `visual/sequence_evidence_map_v1.json`, and
  `visual/viennarna_secondary_structure/` through the bundle manifest.
- folding backend:
  `ViennaRNA` via the uv-managed `viennarna` package, imported as `RNA`, using
  `RNA.fold_compound(...).mfe()` on a T-to-U RNA surrogate mapped back to
  original DNA coordinates
- generated bundle:
  `src/dnadesign/construct/workspaces/retron43_teto_manual_x8/outputs/retron43_teto_manual_x8`
- generated folding prediction:
  `src/dnadesign/construct/workspaces/retron43_teto_manual_x8/outputs/retron43_teto_manual_x8/folding/secondary_structure_prediction_v1.json`
- generated canonical visual contract:
  `src/dnadesign/construct/workspaces/retron43_teto_manual_x8/outputs/retron43_teto_manual_x8/visual/sequence_evidence_map_v1.json`
- generated BaseRender component-span SVG:
  `src/dnadesign/construct/workspaces/retron43_teto_manual_x8/outputs/retron43_teto_manual_x8/visual/renders/component_span_qa_svg/component_span_qa.svg`
- generated ViennaRNA-native structure plot manifest:
  `src/dnadesign/construct/workspaces/retron43_teto_manual_x8/outputs/retron43_teto_manual_x8/visual/viennarna_secondary_structure/viennarna_secondary_structure_svg_v1.json`
- generated ViennaRNA-native annotated SVG:
  `src/dnadesign/construct/workspaces/retron43_teto_manual_x8/outputs/retron43_teto_manual_x8/visual/viennarna_secondary_structure/secondary_structure.annotated.svg`
- generated two-row composition review SVG:
  `src/dnadesign/construct/workspaces/retron43_teto_manual_x8/outputs/retron43_teto_manual_x8/visual/reviews/composition_overview.svg`

Do not commit generated `outputs/` artifacts unless explicitly requested.

### Persistent Alignment Checklist

The full developer spec must remain the normative source for implementation
details. This study page should retain only the Retron-specific handoff facts
that agents need before opening the full spec:

- Construct owns generic `linear_ssdna_composition_v1` execution.
- Retron study records own selected variants and rationale.
- The Retron study owns MSD shorthand parsing and selected-hit metadata through
  `docs/studies/retron_hairpin_design/msd_design_registry.yaml` and the
  study-local module
  `dnadesign.studies.retron_hairpin_design.cli`. Do not expose this as a
  top-level `retron-msd` tool.
- Snapback and scar-nick remain Cruncher primitive lanes.
- Physical sequence pieces are `segment_spans`; overlapping interpretations are
  `annotation_spans`.
- The manual retron-43 literal is treated as an 88 nt unit until a future
  fixture intentionally changes it.
- Scar-nick source refs project only `left_base` and `right_base` into final
  ssDNA by default.
- Type IIS recognition sites, nickase/release footprints, downstream
  degenerate symbols, and upstream processing context are excluded from final
  sequence unless a future public projection contract explicitly selects them.
- Folding is canonical-component-unit and advisory by default. The checked-in
  dogfood route uses the uv-managed ViennaRNA Python API; missing optional CLI
  backends are recorded as `warning_optional_missing` unless the config makes
  folding required.
- Benchling handoff starts with GenBank plus FASTA/CSV sidecars.
  Reverse-complement-derived segments and annotations should appear as
  `complement(...)` GenBank features, with matching `strand`,
  `source_segment_id`, and `transform_kind` columns in `features.csv`.
  Visible GenBank/CSV names should come from the display profile; raw segment
  and annotation ids stay in `dnadesign_*` qualifiers or machine columns.
  Do not duplicate full component spans as same-span annotations.
  Composition CLI output and bundle manifests should expose the generated
  `sequence.gb` path plus an `open -R .../sequence.gb` Finder reveal command
  for local review.
- Retron MSD ID lists should use the study CLI `materialize` route for
  single-unit GenBank/PNG output after concrete payload and cap sequences are
  supplied. It writes `sequence_manifest.json`, `sequence_index.tsv`, generated
  single-unit composition configs, and one `variants/<design-id>/` bundle per
  variant with `sequence.gb`, FASTA/CSV sidecars, and `component_span_qa.png`
  visible at the variant root. If payload or cap sequences are missing, the
  route must fail before generating placeholder GenBank or PNG files. The CLI
  does not expose `--repeat-count`.
- BaseRender consumes only the generated canonical visual contract through
  generated job YAML for linear component-span QA; Construct does not render
  directly.
- `visual/sequence_evidence_map_v1.json` is the canonical component-unit map
  for both BaseRender and ViennaRNA annotation. Do not reintroduce a
  repeat-expanded visual/folding evidence map.
- Component-span QA should keep component color in `span_backdrops` over the
  sequence rows and keep annotation spans out of feature-box fills; the top and
  bottom strands should remain close enough that the solid per-position
  connectors read as a duplex relationship.
- ViennaRNA-native structure plotting is a folding/visual-publisher handoff,
  not a BaseRender layout feature.
- The current ViennaRNA-native dogfood plot records 88 annotated nucleotide
  nodes, 28 basepair nodes, `cross_copy_pair_count=0`,
  `layout_algorithm=naview`, `nucleotide_text_orientation` as
  `upright_counter_rotated`, zero recorded section-label nucleotide,
  reserved-region, and peer-label overlaps, and component hue metadata derived
  from `sequence_evidence_map_v1`. Its annotated title stack reports display
  sections plus canonical `flank_5p`, `payload_primary`,
  `snapback_cap_segment`, `payload_complement`, `flank_3p`, `left_base`, and
  `right_base` terms. Folding QA has a single declared payload
  reverse-complement pairing for the canonical unit and marks it recovered.
- The manual retron-43/TetO dogfood unit intentionally uses the literal
  18 nt `tCCTCAGcccGCTGAGGa` snapback-cap segment. Do not use this fixture to
  infer O33 stem-3/cap-3 geometry; that belongs to the later source-ref
  dogfood slice after fresh Cruncher outputs are rerun and cited.

### Progressive Disclosure

When returning to this work:

1. Read this page for the study-specific boundary.
2. Open the full dev spec:
   [generic linear ssDNA composition](../../dev/plans/2026-05-13-generic-linear-ssdna-composition-spec.md).
3. Use the completed implementation record for historical evidence:
   [generic linear ssDNA composition plan](../../exec-plans/completed/2026-05-13-generic-linear-ssdna-composition.md).
4. Use `routes.md` for the existing Snapback, scar-nick, and YIU command
   routes.
5. Rerun current Cruncher outputs before making exact-hit or PaqCI-capacity
   claims.

### Open Implementation Notes

- Composition belongs in Construct if the contract stays generic.
- Retron-specific selector/ranking logic starts as study config/rationale and
  becomes code only after repeated selection behavior justifies it.
- Folding preflight and ViennaRNA runner behavior now lives in the separate
  `dnadesign.folding` package and is invoked through `uv run folding` or
  Construct's public folding handoff. `RNAfold` is the optional ViennaRNA CLI
  program for requests that select `backend.interface: cli`.
- USR persistence is optional after local artifacts stabilize.
- `msd_design_reference_v1` / `msd_design_catalog_v1` records are the
  pragmatic bridge from study-selected construct IDs to Reader experiments.
  The compact construct label is a human handle; the catalog is the
  source-of-truth contract with profile linting, route metadata, artifact
  pointers, and sequence digests once bundles are attached.
- Ad hoc catalog output should use explicit transient directories such as
  `/tmp/dnadesign_retron_msd_design_references`; the catalog bundle should stay
  shallow, with top-level `README.md`, `manifest.json`,
  `msd_design_catalog_v1.json`, `reference_index.tsv`, and a flat
  `references/` directory. Reader-linked output should be copied into the
  owning Reader experiment `inputs/designs/` directory when that integration is
  implemented. This is deliberately not USR persistence and not workspace
  creation.
- Benchling handoff should start with GenBank plus FASTA/CSV sidecars.
- Naive sequence-artifact Retron MSD requests should start with
  `uv run python -m dnadesign.studies.retron_hairpin_design.cli materialize`
  rather than manually creating Construct workspaces. Provide complete
  subcomponents with `--payload-sequence ID=ACGT`, `--cap-sequence ID=ACGT`,
  and no repeat-count flag; otherwise the compiler reports the missing
  subcomponent and routes back to Snapback or scar-nick.

### Links

- [Retron routes](routes.md)
- [Retron status](status.md)
- [Scar-nick base-junction context](scar-nick-base-junction.md)
- [Generic linear ssDNA composition dev spec](../../dev/plans/2026-05-13-generic-linear-ssdna-composition-spec.md)
- [Generic linear ssDNA composition implementation record](../../exec-plans/completed/2026-05-13-generic-linear-ssdna-composition.md)
- [Linear ssDNA composition hardening follow-ups](../../exec-plans/active/2026-05-14-linear-ssdna-composition-hardening-followups.md)
