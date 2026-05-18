## Exec plan: Generic linear ssDNA composition

**Status:** completed
**Owner:** Shockwing / Codex handoff
**Created:** 2026-05-13
**Last updated:** 2026-05-14
**Authority:** historical implementation record; not an operator router

### Completion Summary

The main local artifact path is implemented through Phase 5: Construct
composition, Benchling-oriented exports, BaseRender component-span QA, Folding
with the uv-managed ViennaRNA Python API, ViennaRNA-native annotated structure
SVG publication, and the two-row composition review.

Open work was split into the active follow-up plan:
[linear ssDNA composition hardening follow-ups](../active/2026-05-14-linear-ssdna-composition-hardening-followups.md).

Progress entries below are implementation history, not the current study phase
and not a reason to default Retron requests into status/preflight.

### Purpose / Big Picture

This plan turns the proposed generic linear ssDNA composition spec into a
progressively checkable implementation path. The immediate dogfood target is
Retron/TetO multicopy ssDNA assembly, but the runtime contract must stay
generic: Construct composes ordered ssDNA segments, Cruncher solves Snapback
and scar-nick primitives, folding emits a backend-neutral prediction contract,
and BaseRender renders visual contracts only.

The implementation is worth doing only if it keeps the current repo posture:
strict schemas, public contracts, fail-fast validation, no sibling `src.*`
imports, and local artifact bundles before optional USR persistence.

### Progress

- [x] (2026-05-13 15:25Z) Persisted the proposed dev spec, Retron study
  handoff, and active execution checklist.
- [x] (2026-05-13 15:33Z) Completed alignment pass against the fuller source
  spec: persisted output contract examples, folding request/result examples,
  span sidecar examples, GenBank mapping, malformed-folding behavior, and
  explicit scar-nick source-ref projection exclusions.
- [x] (2026-05-13 15:55Z) Phase 0: add ADR and contract skeletons for
  `linear_ssdna_composition_v1` and `secondary_structure_prediction_v1`.
- [x] (2026-05-13 15:55Z) Phase 1: implement Construct local composition
  tracer bullet with literal segments, annotations, repeats, reverse-complement
  assertions, and local artifact bundle.
- [x] (2026-05-13 15:55Z) Phase 2: add FASTA, GenBank, and feature CSV export
  for Benchling handoff.
- [x] (2026-05-13 15:55Z) Phase 3a: emit `sequence_evidence_map_v1` from
  composition outputs. BaseRender rendering remains in Phase 3.
- [x] (2026-05-13 16:24Z) Phase 3: enrich `sequence_evidence_map_v1` with
  copy boundaries and intended reverse-complement pairings, emit BaseRender
  component-span QA jobs, and render the retron-43/TetO SVG through BaseRender.
- [x] (2026-05-13 17:31Z) Phase 4: add folding request/result contracts,
  preflight, uv-managed ViennaRNA Python API backend runner, and optional
  ViennaRNA `RNAfold` CLI fallback.
- [x] (2026-05-13 17:41Z) Phase 4 semantics audit: align wording so
  ViennaRNA is the parent package/backend, `RNA` is its Python module, and
  `RNAfold` is its optional CLI program.
- [x] (2026-05-13 18:35Z) Phase 5: add ViennaRNA-native structure SVG
  publishing, dnadesign ontology coloring, and cross-copy pairing summary.
- [x] (2026-05-13 19:05Z) Phase 5 hardening: make ViennaRNA structure plot
  publication explicit through `visual.emit`, pass configured layout selection,
  enrich folding QA with cross-copy and intended-pair recovery summaries, and
  expose the same path through `uv run folding plot`.
- [x] (2026-05-13 19:30Z) Phase 5 information-architecture hardening:
  separated ViennaRNA SVG publishing and pairing QA enrichment from folding
  request/preflight/backend execution so the public folding facade remains
  stable while implementation responsibilities stay explicit.
- [x] (2026-05-13 19:45Z) Phase 3/5 visual semantics hardening: initially
  split repeat-expanded and canonical visual maps. This was superseded by the
  20:20Z correction below because visual and folding QA should not consume
  repeat-expanded product maps in this workflow.
- [x] (2026-05-13 20:20Z) Phase 3/5 visual/folding semantics correction:
  removed repeat-expanded visual/folding evidence. The public
  `visual/sequence_evidence_map_v1.json` is now the canonical 88 nt component
  contract for BaseRender and ViennaRNA annotation, folding uses
  `folding/secondary_structure_input_sequence.json`, and the deprecated
  `visual/contracts/component_span_qa_sequence_evidence_map_v1.json` artifact
  is pruned on regeneration.
- [x] (2026-05-13 20:55Z) Phase 3 component-span visual design hardening:
  moved component color from owner/effect annotation boxes into
  `span_backdrops` that cover both top and bottom strands, enabled one light
  Watson-Crick connector per canonical-unit position, kept annotation labels
  text-only, and regenerated the Retron/TetO SVG output.
- [x] (2026-05-13 21:40Z) Phase 3/5 publication QA hardening: pinned the
  component-span BaseRender handoff to one small display font size, changed
  machine slugs into publication-facing section labels, moved labels closer to
  their strands with collision-safe tier spacing, strengthened component
  backdrops, and added ViennaRNA-native section labels plus cap-right
  orientation metadata/viewBox hardening for annotated structure SVGs.
- [x] (2026-05-13 22:35Z) Phase 3/5 visual artifact hardening: made
  BaseRender duplex connectors solid, embossed the left/right stem-base glyphs
  on both strands, counter-rotated ViennaRNA nucleotide text to remain upright
  after cap-right normalization, reserved title/subtitle space during section
  label placement, added stem-base highlight boxes, and tightened the
  annotated SVG canvas around the final plot.
- [x] (2026-05-13 23:00Z) Phase 5 annotated-plot layout hardening: removed
  ViennaRNA's native white rectangle from the annotated SVG, made the
  dnadesign background the single fitted canvas, centered title/subtitle text
  on the normalized hairpin content, added canonical component,
  `snapback_foldback_geometry`, and `scar_nick` `left_base`/`right_base` subtitle
  lines, and added peer-label collision accounting.
- [x] (2026-05-14 00:05Z) Phase 5 composition-review design pass: moved
  secondary-structure summary wording into a dedicated folding helper, shortened
  the annotated SVG title/subtitle to publication-facing values, and added a
  Construct-owned two-row `composition_review_svg_v1` publisher that stacks the
  ViennaRNA annotated structure over the BaseRender component-span SVG.
- [x] (2026-05-14 00:30Z) Phase 5 composition-review scale correction:
  changed the two-row overview from literal nucleotide-font matching to a
  `balanced_visual_weight` policy so the ViennaRNA structure and BaseRender
  component span read as comparable subplots. The current dogfood target scales
  the component-span row beyond strict width matching and applies review-only
  glyph emphasis instead of leaving it as an undersized comparison strip.
- [x] (2026-05-14 01:05Z) Phase 5 review balance and stem-base annotation
  hardening: added a default-on `emphasize_stem_base_nucleotides` plot option,
  propagated it through Construct and the folding CLI, tightened left/right
  stem-base label placement around the annotated ViennaRNA structure, and
  recorded visual-weight balance plus effective nucleotide size in
  `composition_review_svg_v1`.
- [x] (2026-05-14 01:40Z) Phase 5 typography and redundant-title hardening:
  pinned ViennaRNA section labels and subtitles to the same annotation font
  size, removed the standalone BaseRender component-span title only when
  embedding that SVG in the compound review, and recorded the omission policy
  and count in `composition_review_svg_v1`.
- [x] (2026-05-14 02:12Z) Phase 5 information-architecture hardening:
  added manifest-backed `uv run folding preflight|run|plot --bundle <bundle>`
  commands so operators can target Construct bundles without spelling every
  nested artifact path, while keeping Folding workspace-less and producer-owned
  outputs in place.
- [x] (2026-05-14 03:02Z) Study-owned MSD reference compiler slice: added
  `msd_design_reference_v1` / `msd_design_catalog_v1`, strict Retron MSD label
  parsing with scar-nick `S3/S2/S1/S0` profile linting, checked-in registry
  entries for the scar-nick hit list, and a study-local CLI module under
  `dnadesign.studies.studies.retron_hairpin_design` without adding a top-level
  `retron-msd` script.
- [x] (2026-05-14 03:40Z) Study information-architecture hardening: added the
  selected MSD label list, exposed the ID-to-catalog route through
  `pipeline.yaml`, `ops.study.yaml`, route docs, and the retron-hairpin skill
  references, and recorded the product posture as user-provided parts plus
  study registry metadata compiled into transient design-reference catalogs
  rather than Construct/Folding workspace sprawl.
- [x] (2026-05-14 01:45Z) Phase 7 partial scar-nick source-output pressure
  test: regenerated same-workspace BbsI-HF and PaqCI `scar_nick_teto` bundles,
  rendered their BaseRender terminal-nick QA PNGs, added a PaqCI core-panel
  spec, and recorded that current strict public catalogs cover 13/14 active
  profile buckets with `WMWM` still uncovered.
- [x] (2026-05-14 00:00Z) Deferred Phase 6 optional USR persistence and
  remaining Phase 7 source-ref dogfood to the active follow-up plan so this
  implementation record can close without carrying current-phase work.

### Surprises & Discoveries

The retron-43 literal sequence currently resolves to an 88 nt unit under
zero-based half-open coordinates. Earlier shorthand span sketches used an 87 nt
layout. Implementation must compute spans from segment lengths and make the
coordinate reconciliation a golden-test fixture.

Scar-nick handoff needs a projection rule, not raw context import. The
terminal-nick feasibility model may carry Type IIS recognition sites, nickase
footprints, downstream degenerate symbols, protected/discarded strand burden,
and visual context. For final linear ssDNA composition, only the four-base
`left_base` and `right_base` spans are projected into the final product unless
a future public contract explicitly selects more sequence.

The first persisted spec was directionally complete but too compressed for
agent handoff in the contract sections. The dev spec now carries enough
copyable examples for an implementation agent to write fixtures without
reconstructing the prior chat context.

The first implementation slice kept FASTA sidecars unwrapped. That keeps the
manual retron-43 88 nt unit inspectable in the checked-in dogfood fixture while
JSON remains the source of truth for exact spans and sequence digest.

The BaseRender handoff is a generated job YAML, not a Construct-to-BaseRender
runtime import. Construct writes `baserender_jobs/component_span_qa_<fmt>.yaml`
inside the local artifact bundle. That job consumes the canonical component QA
contract at `visual/sequence_evidence_map_v1.json`; the same canonical contract
is used for ViennaRNA annotation.

The current output tree remains conservative for compatibility. `folding/` and
`visual/viennarna_secondary_structure/` are semantically clear, but
`visual/renders/component_span_qa_svg/component_span_qa.svg` is more nested
than ideal because BaseRender writes non-workspace job outputs under
`results_root/<job_stem>/`. A future path-compatibility slice should either
teach BaseRender a first-class flat output mode or rename the job stem to a
format-neutral `component_span_qa`; do not hand-move generated outputs.

Folding now accepts producer-owned bundles directly through `--bundle` for
preflight, run, and plot commands. Construct's bundle manifest is one supported
producer contract, not a Folding storage model. This is an operator-surface
cleanup, not a new storage root: Folding still has no workspace, reads
bundle-owned manifest entries, and writes plot artifacts under the same bundle's
`visual/viennarna_secondary_structure/` directory.

The system `RNAfold` executable is not on the local PATH, but the official
ViennaRNA Python package is now uv-managed as `viennarna==2.7.2` and imports as
`RNA`. The dogfood config uses `backend.interface: python_api` and
`RNA.fold_compound(...).mfe()`, so folding now records `status=ok` without
depending on a separately installed executable. The CLI interface remains
available as an explicit optional fallback and still reports
`warning_optional_missing` when configured as advisory and missing.

Semantic audit note: official ViennaRNA sources frame `RNAfold` as one
command-line program shipped by the ViennaRNA Package and `RNA` as the Python
module/interface to the same package. Repo wording should therefore say
`ViennaRNA` for the backend package and use `RNAfold` only when referring to
that specific CLI executable or its stdout-compatible format.

ViennaRNA plotting audit note: ViennaRNA already owns the secondary-structure
layout problem through `RNAplot` and Python API plotting helpers such as
`RNA.svg_rna_plot` and `RNA.plot_structure_svg`. Phase 5 must wrap and annotate
native ViennaRNA SVG output rather than recreate fold-layout primitives in
BaseRender. BaseRender remains the linear component-span QA renderer.

Visual artifact audit note: the checked-in manual retron-43/TetO fixture uses
the literal `tCCTCAGcccGCTGAGGa` snapback-cap segment, which is 18 nt. Do not
read this artifact as an O33 stem-3/cap-3 geometry claim. Exact O33 cap/stem
semantics belong in a fresh source-ref dogfood slice after current Cruncher
outputs are rerun and referenced.

Scar-nick source-output audit note: the current `scar_nick_teto` workspace
does not need one workspace per retained-scar hit. Multiple release-enzyme
specs live under `configs/scar_nick/` and write separate generated run dirs
under `outputs/scar_nick/`. The BbsI-HF and PaqCI specs now validate and render
in that single workspace. BbsI-HF covers the 10-bucket historical panel; PaqCI
adds `WXMM`, `WMMM`, and `WWMM`, bringing combined strict-catalog coverage to
13/14 active buckets. `WMWM` remains absent under the exact-terminal-nick,
downstream-degenerate, `S0=M` public-catalog policy and should not be described
as a generated hit until a fresh run proves otherwise.

Annotated ViennaRNA plot layout note: the annotated SVG should have exactly
one dnadesign-owned white background rectangle fitted to the final viewBox.
The native ViennaRNA background rectangle remains in
`secondary_structure.native.svg`, but not in `secondary_structure.annotated.svg`.
The title/subtitle block is centered on the normalized hairpin geometry. Keep
the visible wording short and publication-facing: for the manual retron-43/TetO
dogfood this means `Retron 43 TetO x8`, `TetO payload | left CAAG / right CTCG`,
and `Foldback tCCTCAGcccGCTGAGGa (18 nt)` plus `Cap ccc (3 nt)`, not machine-key subtitle lines.

Composition review note: the two-row overview is Construct-owned because it
combines one folding artifact and one BaseRender artifact. It should live under
`visual/reviews/`, not under the ViennaRNA or BaseRender renderer-owned output
directories. The current policy is `balanced_visual_weight`: keep the folded
structure readable while scaling and emboldening the component-span row enough
that the lower subplot is a comparable visual peer rather than a thin
comparison strip. The lower row should not carry the standalone component-span
title inside the compound overview because the top title/subtitle already
identifies the composition. Annotation text in the ViennaRNA plot should use a
single pinned size for section labels and subtitles.

### Decision Log

Decision: Composition authority belongs in Construct, but only as generic
linear ssDNA composition.

Rationale: Construct is the package boundary for deterministic sequence
assembly. Retron biology belongs in the study record or later study-owned
selector code.

Date/Author: 2026-05-13 / Shockwing + Codex

Decision: The first slice writes local artifact bundles, not mandatory USR
rows or overlays.

Rationale: The span and annotation contract should stabilize before durable
USR schema and overlay behavior are added.

Date/Author: 2026-05-13 / Shockwing + Codex

Decision: Folding is advisory by default and separate from BaseRender.

Rationale: Folding is scientific backend execution; BaseRender should render
typed visual contracts and should not run ViennaRNA interfaces directly.

Date/Author: 2026-05-13 / Shockwing + Codex

Decision: Scar-nick source refs project only final sequence spans by default.

Rationale: Type IIS recognition sequence and upstream processing context are
cloning/provenance/feasibility context. The final multicopy ssDNA insert uses
only the four-base left/right basal spans plus separately selected payload,
cap, complement, and flank segments.

Date/Author: 2026-05-13 / Shockwing + Codex

Decision: Construct publishes BaseRender job handoffs instead of invoking
BaseRender.

Rationale: This keeps Construct responsible for sequence assembly and visual
contract publication while BaseRender remains the only renderer. The generated
job is deterministic, local to the artifact bundle, and can be validated or run
through the public BaseRender CLI/API.

Date/Author: 2026-05-13 / Shockwing + Codex

Decision: Folding execution is a separate `dnadesign.folding` package with a
public API and `uv run folding` CLI. The default dogfood backend is the
uv-managed ViennaRNA Python API, while system-provided ViennaRNA `RNAfold` CLI
execution remains an optional fallback interface.

Rationale: Construct may write and submit a typed folding request through the
public folding API, but ViennaRNA execution, preflight, parsing, and
explicit degraded states belong outside Construct and outside BaseRender. The
official Python interface is lockfile-managed through uv for reproducible local
dogfooding; the runner can still use a system-provided ViennaRNA `RNAfold`
executable when a request chooses `backend.interface: cli`.

Date/Author: 2026-05-13 / Shockwing + Codex

### Outcomes & Retrospective

Phase 0 through Phase 5 landed as TDD-backed Construct, BaseRender handoff,
folding-runner, uv-managed ViennaRNA Python API, and ViennaRNA-native SVG
publisher slices.

Implemented artifacts:

- ADR:
  `docs/architecture/decisions/adr-0002-generic-linear-ssdna-composition.md`
- Contracts:
  `src/dnadesign/contracts/sequence/linear_ssdna_composition_v1.py`
  and `src/dnadesign/contracts/folding/secondary_structure_prediction_v1.py`
- Construct runtime:
  `src/dnadesign/construct/src/composition.py`
- CLI:
  `uv run construct compose validate --config <composition.yaml>`
  and `uv run construct compose run --config <composition.yaml>`
- Folding runtime and CLI:
  `src/dnadesign/folding/` with
  `uv run folding preflight --request <request.yaml>` and
  `uv run folding run --request <request.yaml>`
- Folding dependency:
  `viennarna>=2.7.2`, imported as `RNA` by the Python API backend
- Dogfood config:
  `src/dnadesign/construct/workspaces/retron43_teto_manual_x8/config.composition.yaml`
- Generated BaseRender handoff:
  `baserender_jobs/component_span_qa_svg.yaml` inside each local composition
  artifact bundle
- Generated folding handoff:
  `folding/secondary_structure_prediction_request_v1.yaml`,
  `folding/folding_preflight.json`, and
  `folding/secondary_structure_prediction_v1.json` inside each local
  composition artifact bundle when folding is enabled
- Generated ViennaRNA-native structure plot handoff:
  `visual/viennarna_secondary_structure/viennarna_secondary_structure_svg_v1.json`,
  `visual/viennarna_secondary_structure/secondary_structure.native.svg`,
  `visual/viennarna_secondary_structure/secondary_structure.annotated.svg`,
  and
  `visual/viennarna_secondary_structure/secondary_structure.annotation_manifest.json`
  when folding succeeds

Validation evidence:

- RED: targeted tests first failed on missing
  `dnadesign.contracts.folding` and `dnadesign.construct.src.composition`.
- GREEN:
  `uv run pytest -q src/dnadesign/contracts/tests/test_sequence_contracts.py src/dnadesign/construct/tests/runtime/test_linear_ssdna_composition.py src/dnadesign/construct/tests/cli/test_compose_command.py`
  passed with 8 tests.
- WIDER:
  `uv run pytest -q src/dnadesign/construct/tests src/dnadesign/contracts/tests`
  passed.
- DOGFOOD:
  `uv run construct compose validate --config src/dnadesign/construct/workspaces/retron43_teto_manual_x8/config.composition.yaml --format json`
  returned `sequence_length=704`.
- PHASE 3 RED:
  `uv run pytest -q src/dnadesign/construct/tests/runtime/test_linear_ssdna_composition.py`
  failed because the visual contract lacked copy boundaries/pairings and the
  manifest lacked `baserender_component_span_svg_job`.
- PHASE 3 GREEN:
  `uv run pytest -q src/dnadesign/construct/tests/runtime/test_linear_ssdna_composition.py`
  passed with 3 tests.
- PHASE 3 SCOPED:
  `uv run pytest -q src/dnadesign/construct/tests src/dnadesign/contracts/tests`
  passed.
- PHASE 3 BASERENDER SMOKE:
  `uv run pytest -q src/dnadesign/baserender/tests/test_yiu_contract_jobs.py::test_run_job_renders_sequence_evidence_map_contract src/dnadesign/baserender/tests/test_adapter_registry.py::test_sequence_evidence_map_adapter_applies_contract_without_complement_sequence`
  passed.
- PHASE 3 LINT:
  `uv run ruff check ...` and `uv run ruff format --check ...` passed for the
  composition-slice Python files.
- PHASE 3 DOGFOOD:
  `uv run construct compose run --config src/dnadesign/construct/workspaces/retron43_teto_manual_x8/config.composition.yaml --format json`
  returned `sequence_length=704`, then
  `uv run baserender job run src/dnadesign/construct/workspaces/retron43_teto_manual_x8/outputs/retron43_teto_manual_x8/baserender_jobs/component_span_qa_svg.yaml`
  wrote `visual/renders/component_span_qa_svg/component_span_qa.svg`.
- PHASE 4 RED:
  `uv run pytest -q src/dnadesign/contracts/tests/test_sequence_contracts.py src/dnadesign/folding/tests src/dnadesign/construct/tests/runtime/test_linear_ssdna_composition.py`
  failed on missing `SecondaryStructurePredictionRequestV1` and missing
  `dnadesign.folding`.
- PHASE 4 GREEN:
  `uv run pytest -q src/dnadesign/contracts/tests/test_sequence_contracts.py src/dnadesign/folding/tests src/dnadesign/construct/tests/runtime/test_linear_ssdna_composition.py`
  passed with 14 tests.
- PHASE 4B RED:
  the same target suite failed after adding Python API expectations because
  `SecondaryStructurePredictionRequestV1` and
  `LinearSsdnaCompositionV1` did not yet accept `backend.interface` or
  `backend.python_module`.
- PHASE 4B GREEN:
  `uv run pytest -q src/dnadesign/contracts/tests/test_sequence_contracts.py src/dnadesign/folding/tests/test_rnafold_runner.py src/dnadesign/construct/tests/runtime/test_linear_ssdna_composition.py`
  passed with 16 tests after adding the uv-managed ViennaRNA Python API
  backend path.
- PHASE 4 SCOPED:
  `uv run pytest -q src/dnadesign/construct/tests src/dnadesign/contracts/tests src/dnadesign/folding/tests`
  passed.
- PHASE 4 CLI:
  `uv run folding --help` exposed `preflight` and `run`.
- PHASE 4 DOGFOOD:
  `uv run construct compose run --config src/dnadesign/construct/workspaces/retron43_teto_manual_x8/config.composition.yaml --format json`
  returned `sequence_length=704` and wrote folding artifacts through the
  uv-managed ViennaRNA Python API. The system `RNAfold` executable, which is
  the ViennaRNA CLI program, is still not installed locally, but this request
  does not need it:
  `uv run folding preflight --request src/dnadesign/construct/workspaces/retron43_teto_manual_x8/outputs/retron43_teto_manual_x8/folding/secondary_structure_prediction_request_v1.yaml --format json`
  returned `status=ok`, `backend.name=ViennaRNA`, `backend.version=2.7.2`,
  `backend.interface=python_api`, and no resolved executable. The prediction
  artifact now uses the canonical component-unit sequence digest
  `60152ddb90ec78bf43a184b92562f5d983eba8a908cceec7b0ddfbb833443315`,
  dot-bracket length `88`, MFE `-53.7`, and `28` parsed base pairs.
- PHASE 4 LINT:
  `uv run ruff check ...` and `uv run ruff format --check ...` passed for the
  composition/folding/contracts slice.
- PHASE 4 BOUNDARY:
  `uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .`
  passed after registering `folding` as an explicit top-level tool boundary
  and allowing Construct to call its public API.
- PHASE 5 RED:
  `uv run pytest -q src/dnadesign/folding/tests/test_rnafold_runner.py src/dnadesign/construct/tests/runtime/test_linear_ssdna_composition.py`
  failed on missing `publish_viennarna_structure_svg`.
- PHASE 5 GREEN:
  the same target passed after adding the public publisher, the
  `viennarna_secondary_structure_svg_v1` manifest contract, SVG DOM annotation,
  and Construct handoff.
- PHASE 5 SCOPED:
  `uv run pytest -q src/dnadesign/contracts/tests src/dnadesign/folding/tests src/dnadesign/construct/tests`
  passed.
- PHASE 5 DOGFOOD:
  `uv run construct compose run --config src/dnadesign/construct/workspaces/retron43_teto_manual_x8/config.composition.yaml --format json`
  returned `sequence_length=704` and wrote the ViennaRNA-native structure plot
  bundle. The generated `viennarna_secondary_structure_svg_v1` manifest records
  `backend_version=2.7.2`, `layout_algorithm=naview`,
  `nucleotide_node_count=88`, `basepair_node_count=28`,
  `cross_copy_pair_count=0`, and no warnings or errors. Annotation manifest
  coordinate `index_0=15` maps to
  `display_index_1=16`, `owner_ids=["retron43_teto_unit.payload_primary"]`,
  `effect_tags=[]`, and hue `#F58518`.
- PHASE 5 BASERENDER ORTHOGONAL VIEW:
  `uv run baserender job run src/dnadesign/construct/workspaces/retron43_teto_manual_x8/outputs/retron43_teto_manual_x8/baserender_jobs/component_span_qa_svg.yaml`
  still wrote the separate linear component-span SVG under
  `visual/renders/component_span_qa_svg/`.
- PHASE 5 LINT:
  slice Ruff check, slice Ruff format check, docs check, architecture boundary
  check, and `git diff --check` passed.
- PHASE 5 HARDENING RED:
  targeted tests failed on missing `enrich_prediction_pairing_qa` and then on
  missing `uv run folding plot`.
- PHASE 5 HARDENING GREEN:
  `uv run pytest -q src/dnadesign/contracts/tests/test_sequence_contracts.py src/dnadesign/folding/tests src/dnadesign/construct/tests/runtime/test_linear_ssdna_composition.py`
  passed with 21 tests after adding explicit plot opt-in, configured
  ViennaRNA layout propagation, folding QA pairing summaries, intended-pair
  SVG metadata, and the folding plot CLI.
- PHASE 5 HARDENING DOGFOOD:
  `uv run construct compose run --config src/dnadesign/construct/workspaces/retron43_teto_manual_x8/config.composition.yaml --format json`
  returned `sequence_length=704` with the same sequence digest. The enriched
  folding QA records `predicted_pair_count=28`, `cross_copy_pair_count=0`,
  `intended_pairing_count=1`, `intended_recovered_count=1`, and
  `intended_missed_count=0`. The plot CLI path
  `uv run folding plot --prediction ... --assembled-sequence ... --visual-contract ... --output-dir ../visual/viennarna_secondary_structure --layout naview`
  also published the dogfood manifest successfully.
- PHASE 5 IA HARDENING:
  `src/dnadesign/folding/src/api.py` no longer owns ViennaRNA SVG DOM
  annotation or pairing recovery internals; the public API and CLI behavior are
  unchanged.
- PHASE 5 MAINTAINABILITY HARDENING:
  the ViennaRNA plot publisher was split by responsibility so
  `src/dnadesign/folding/src/viennarna_plot.py` coordinates artifact emission,
  `src/dnadesign/folding/src/viennarna_svg.py` owns SVG DOM annotation and
  cap-right layout normalization, `src/dnadesign/folding/src/pairing_qa.py`
  owns backend-neutral pairing QA, and
  `src/dnadesign/folding/src/viennarna_ontology.py` owns component hue/slug
  semantics. The publisher module is no longer a monolithic SVG/pairing
  implementation module.
- PHASE 5 OUTPUT-IA HARDENING:
  `uv run folding plot` now resolves plain relative `--output-dir` paths from
  the current working directory, while `../...` paths remain bundle-relative to
  the prediction artifact directory. This prevents repo-root style paths such
  as `src/dnadesign/.../visual/viennarna_secondary_structure` from being
  silently nested under `folding/src/...`.
  Construct also removes the deprecated generated `folding/src` tree on bundle
  regeneration.
- PHASE 5 MAINTAINABILITY VALIDATION:
  `uv run pytest -q src/dnadesign/contracts/tests src/dnadesign/folding/tests src/dnadesign/construct/tests/runtime/test_linear_ssdna_composition.py src/dnadesign/construct/tests/cli/test_compose_command.py src/dnadesign/baserender/tests/test_sequence_evidence_component_view.py`
  passed. `uv run construct compose validate ...`, `uv run construct compose
  run ...`, `uv run baserender job run ...`, and `uv run folding plot ...`
  passed for the retron-43/TetO dogfood bundle. `uv run ruff check .`,
  `uv run ruff format --check .`, docs checks, architecture boundary checks,
  and `git diff --check` passed.
- PHASE 5 REPO-WIDE RESIDUAL:
  full `uv run pytest -q` still fails outside this slice in LatentDNA/stress
  study drift:
  `test_live_study_recipes_rebuild_from_clean_workspace_state`,
  `test_display_hue_label_names_sfxi_reference_metric`, and
  `test_latentdna_readme_routes_to_reference_first_docs`.
- DOCS:
  `uv run python -m dnadesign.devtools.docs.checks --repo-root .` passed
  with 297 markdown files.
- REPO-WIDE RESIDUAL:
  `uv run ruff check .` is blocked by unrelated dirty `latentdna` issues:
  `src/dnadesign/latentdna/src/scalars/build.py` has a line-length/formatting
  issue and `src/dnadesign/latentdna/src/sources/infer_sidecar_join.py` still
  has the unused `wanted_keys` variable. `uv run ruff format --check .` is
  blocked by the same unrelated `latentdna/src/scalars/build.py` formatting
  drift.
  `uv run pytest -q` is blocked outside this slice by unrelated LatentDNA
  live-study recipe expectation drift and existing USR layout inventory drift
  for tracked `maintenance.py` and `test_sequence_view_alias_repair.py` files.

The generated local artifact bundle is intentionally not checked in. It can be
regenerated from the dogfood config when needed.

### Context and Orientation

Read these files first:

- Dev spec:
  [generic linear ssDNA composition spec](../../dev/plans/cross-tool/linear-ssdna-composition/2026-05-13-generic-linear-ssdna-composition.md)
- Study handoff:
  [Retron linear ssDNA composition](../../studies/retron_hairpin_design/contexts/composition/linear-ssdna-composition.md)
- Study routes:
  [Retron Hairpin routes](../../studies/retron_hairpin_design/routes/README.md)
- Scar-nick context:
  [scar-nick base-junction](../../studies/retron_hairpin_design/contexts/cruncher/scar-nick-base-junction.md)
- Planning lifecycle:
  [PLANS](../../../PLANS.md)

Current boundaries to preserve:

- Construct composes generic sequence products.
- Cruncher Snapback solves cap/shortening candidates.
- Cruncher scar_nick solves Type IIS scar plus terminal nick feasibility.
- YIU remains contrast-only.
- Folding emits typed prediction artifacts.
- BaseRender consumes visual contracts only.
- Study records carry Retron-specific rationale and dogfooding links.

### Plan of Work

Phase 0: ADR and contract skeleton.

Add the ADR and strict shared schemas. Keep runtime behavior minimal. Tests
should exercise valid and invalid fixtures, unsupported versions, duplicate
IDs, and round trips.

Phase 1: Construct local composition tracer bullet.

Add a public Construct path for literal segment composition with repeats,
annotations, reverse-complement assertions, validation reports, and local JSON
artifacts. The manual retron-43/TetO example and a tiny synthetic fixture are
the first fixtures.

Phase 2: Benchling exports.

Add FASTA, GenBank, and feature CSV sidecars. Internal coordinates remain
zero-based half-open; GenBank exports convert to one-based inclusive feature
locations.

Phase 3: Visual contract publisher.

Emit `sequence_evidence_map_v1` from composition outputs as the canonical
representative component-unit contract. Do not emit a repeat-expanded visual
contract for folding, ViennaRNA annotation, or BaseRender component-span QA.
BaseRender must not import Construct internals.

Phase 4: Folding backend.

Add a backend-neutral folding request/result contract plus ViennaRNA Python API
and ViennaRNA `RNAfold` CLI preflight/parser support. Folding runs on the
canonical component-unit sequence, records explicit DNA/RNA policy, and is
advisory unless configured required.

Phase 5: ViennaRNA-native folding visual QA.

Publish native ViennaRNA structure SVG from composition plus folding output.
Validate the observed ViennaRNA SVG surface, post-process nucleotide and
base-pair nodes with dnadesign coordinate/ontology metadata and component
hues, and emit a `viennarna_secondary_structure_svg_v1` manifest. Do not build
a secondary-structure layout engine in BaseRender; keep BaseRender on the
orthogonal linear component-span QA view.

Deferred Phase 6: Optional USR persistence.

After local artifacts stabilize, add opt-in USR write behavior with conflict
guards, digest refs, and only the overlays that carry concrete value.

Deferred Phase 7: Study dogfooding and source refs.

Link manual retron-43/TetO output into the study record. Add optional `de033`
and `scar_nick_teto` source-ref demo only after rerunning current outputs that
support any exact-hit or route-capacity claims.

### Concrete Steps

1. Add ADR `docs/architecture/decisions/adr-0002-generic-linear-ssdna-composition.md`.
2. Add contract fixtures for minimal synthetic and retron-43 compositions.
3. Add `linear_ssdna_composition_v1` under the chosen contracts namespace.
4. Add `secondary_structure_prediction_v1` under the chosen folding/contracts
   namespace.
5. Add schema round-trip and fail-fast tests.
6. Add Construct parser/validator and local composer.
7. Add artifact writers for assembled sequence, segment spans, annotation
   spans, provenance, and validation report.
8. Add retron-43 golden test with 88 nt literal-coordinate reconciliation.
9. Add scar-nick projection tests asserting Type IIS/protected context is not
   concatenated into final ssDNA.
10. Add FASTA, GenBank, and feature CSV exports.
11. Emit one canonical component-unit `sequence_evidence_map_v1` artifact and
    wire the BaseRender render path to that contract.
12. Add folding preflight, ViennaRNA backend runner, parser, and prediction
    contract.
13. Add folding visual contract and renderer smoke.
14. Add optional USR write path only after local artifacts are stable.
15. Update this execution plan's `Progress`, `Surprises & Discoveries`, and
   `Decision Log` after each implementation slice.
16. During Phase 0, verify the spec examples become checked-in fixtures or
    fixture-adjacent docs so later tests cannot drift from the documented
    retron-43 coordinates and scar-nick projection rule.

Completed first slice:

- 1. Added the ADR.
- 2. Added contract tests for minimal synthetic and retron-43 compositions.
- 3. Added `linear_ssdna_composition_v1` under `contracts/sequence`.
- 4. Added `secondary_structure_prediction_v1` under `contracts/folding`.
- 5. Added schema and fail-fast tests.
- 6. Added Construct parser/validator and local composer.
- 7. Added artifact writers for assembled sequence, segment spans, annotation
  spans, provenance, validation report, FASTA, GenBank, feature CSV, manifest,
  and visual contract JSON.
- 8. Added retron-43 golden test with 88 nt literal-coordinate reconciliation.
- 10. Added FASTA, GenBank, and feature CSV exports.
- 12. Added folding preflight, ViennaRNA Python API and ViennaRNA `RNAfold`
  CLI runners, parser, and prediction contract integration.
- 16. Added a checked-in dogfood composition config under Construct workspaces.

### Validation and Acceptance

Functional acceptance:

- Manual retron-43/TetO composition assembles from literal segments.
- Physical segments are contiguous, non-overlapping, and cover each unit copy.
- Semantic annotations may overlap and are bounds-checked.
- Payload reverse-complement assertion passes.
- Repeats expand deterministically with copy-level spans.
- Scar-nick source refs project only four-base left/right final sequence spans
  unless a future explicit projection contract says otherwise.
- Type IIS recognition sequence, nickase footprint, and upstream processing
  context do not enter the final ssDNA output by default.
- GenBank feature locations convert correctly from zero-based half-open spans.
- Visual contract validates and renders without private cross-tool imports.
- Folding result records backend/version/parameters/DNA policy and validates
  dot-bracket length and pair map bounds.

Repo acceptance:

- `uv run ruff check .` passes.
- `uv run ruff format --check .` passes.
- `uv run pytest -q` passes or unrelated pre-existing failures are documented.
- `uv run python -m dnadesign.devtools.docs.checks --repo-root .` passes.
- Architecture-boundary checks continue to reject sibling internal imports.
- No generated artifacts are committed unintentionally.

### Links

- Proposal:
  [generic linear ssDNA composition spec](../../dev/plans/cross-tool/linear-ssdna-composition/2026-05-13-generic-linear-ssdna-composition.md)
- Study handoff:
  [Retron linear ssDNA composition](../../studies/retron_hairpin_design/contexts/composition/linear-ssdna-composition.md)
- PR: https://github.com/e-south/dnadesign/pull/47
- ADR:
  [ADR 0002: Generic linear ssDNA composition in Construct](../../architecture/decisions/adr-0002-generic-linear-ssdna-composition.md)
- Follow-up plan:
  [linear ssDNA composition hardening follow-ups](../active/2026-05-14-linear-ssdna-composition-hardening-followups.md)
