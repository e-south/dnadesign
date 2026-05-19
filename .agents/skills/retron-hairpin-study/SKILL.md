---
name: retron-hairpin-study
description: Route Retron MSD compiler work. Use for MSD IDs, sequence bundles, design catalogs, GenBank/native-structure PNG outputs, Finder opens, or missing MSD parts. Do not use for generic Cruncher/snapback.
metadata:
  version: 0.7.12
  category: workflow-automation
  tags: [retron, msd, genetic-compiler, snapback, scar-nick, composition, study]
---

# Retron Hairpin Study

## Purpose

Route Retron MSD work as a genetic compiler: compile a reference, materialize one MSD unit, or route missing constraints to a primitive solver.

## Scope

In scope:
- Retron MSD shorthand IDs and explicit part sets.
- Typed `retron_msd_compiler_spec_v1` files with labels, explicit designs, and selected public primitive sources.
- Study-owned `msd_design_reference_v1` / `msd_design_catalog_v1` compilation and workbench provenance.
- Routing missing cap/shortening constraints to Snapback.
- Routing missing stem-base or terminal-nick constraints to scar-nick base-junction.
- Routing mismatch-display questions to YIU as contrast only.
- Construct/BaseRender service calls for one MSD unit per design after parts are selected.
- Skill, route-map, and compiler-harness hardening.

Out of scope:
- Generic Cruncher walkthroughs.
- Bench-level retron protocol advice.
- Making Retron MSD compilation a top-level `retron-msd` tool.
- Creating one Construct or Folding workspace per requested design.
- Reporting phase/status posture unless the user explicitly asks for study progress or blockers.

## Success Criteria

- The first decision is input completeness: compile now, or route missing constraints.
- Complete user-provided parts are validated and compiled without solver work.
- Compiler specs are parsed at the boundary, then compile from trusted part structures.
- Incomplete parts route to the smallest primitive: Snapback, scar-nick, or YIU contrast.
- Solved primitive inputs come through public Snapback/scar-nick APIs; selectors must be explicit and multi-option selections must not expand silently.
- Sequence artifact output is one MSD unit per design: 5' flank + left base,
  payload primary, user-selected cap/foldback segment, payload complement,
  right base + 3' flank. Snapback subsection annotations are emitted only when
  topology is supplied.
- Materialized `msd_design_id` / variant directory names must preserve the cap/base/profile ontology in filenames, using `C172-LCGGT-RACAG-MXMM` style suffixes.
- Requests for "outputs", "deliverables", "exports", "GenBank", "plots", or
  "open in Finder" must run `materialize`; a reference catalog is not enough.
- Materialized plot deliverables require ViennaRNA status `ok`; publish
  `secondary_structure.native.png`, two-row `composition_overview.svg`, and
  high-resolution `composition_overview.png`, not legacy composites.
- Secondary-structure subtitles must include the scar-nick mismatch profile
  from the selected MSD design, for example `mismatch profile MXMM`.
- No user-facing repeat count; do not chain complete MSD units together.
- GenBank/CSV output uses display labels, keeps raw ids in machine qualifiers, and avoids duplicate full component spans as same-span annotations.
- Persistent hypotheses/design-set meaning lives in `workbench/`; generated
  outputs go to explicit transient or caller-owned directories.
- Default `S0=M` is required. Profile drift, non-ligatable S0 labels without explicit control opt-in, unknown registry parts, and missing artifacts fail fast. Deliberate controls require `--allow-non-ligatable-s0` or `allow_non_ligatable_s0: true`, and emitted references must carry `scar_nick.s0_match_required=false`.
- Status/preflight commands are optional progress tools, not default answer
  posture.

## Workflow

1. Classify the request.
- Complete MSD label or complete parts: use [msd-design-references.md](references/msd-design-references.md).
- Complete MSD labels plus "outputs", "deliverables", "exports", "plots",
  "GenBank", or "open in Finder": materialize, not compile-only.
- Typed compiler spec: lint with `--spec`; accept labels or explicit designs,
  and use `selector.mode=rank` for the preferred explicit primitive
  combination.
- Need sequence, visual, or GenBank: materialize with `--spec` or explicit
  payload/cap sequences; `C###` cap IDs never imply a de033 sequence by pattern.
- Need an intentional non-ligatable S0 control: materialize with `--allow-non-ligatable-s0` or a typed spec that sets `allow_non_ligatable_s0: true`; do not use this for profile-drift errors.
- Need hypotheses, effect tags, design sets, or run provenance: open `docs/studies/retron_hairpin_design/workbench/`.
- Missing cap, shortening, or stem/cap geometry: route to Snapback in
  `docs/studies/retron_hairpin_design/routes/README.md`.
- Missing left/right base feasibility, terminal-nick route, nickase, or
  `S3/S2/S1/S0` profile: route to scar-nick.
- Mismatch-only display or boundary contrast: route to YIU; it is not the
  topology engine.
- Progress or blocker question: only then use
  `uv run ops progress show studies.retron-hairpin-design.status --study-dir docs/studies/retron_hairpin_design --json` or
  `uv run ops progress show studies.retron-hairpin-design.preflight --study-dir docs/studies/retron_hairpin_design --scope next --json`.

2. Load only the needed surfaces.
- Compiler route: `docs/studies/retron_hairpin_design/routes/README.md`, then
  `docs/studies/retron_hairpin_design/compiler/catalog/msd_design_registry.yaml`, then
  `references/msd-design-references.md`.
- Whole-product context: `docs/studies/retron_hairpin_design/contexts/composition/linear-ssdna-composition.md` plus the active exec plan.
- Machine-readable command groups: open `docs/studies/retron_hairpin_design/operations/runtime/command-groups/README.md`
  first, then the matching `command-groups/lanes/` sidecar; use `pipeline.yaml` only for the full payload.
- Ownership boundaries: [study-surfaces.md](references/study-surfaces.md).

3. Execute or report the route.
- For complete reference inputs, run `uv run python -m dnadesign.studies.studies.retron_hairpin_design.interfaces.cli.app lint|compile`.
- For GenBank/structure-review output, run the same module's `materialize` command with
  `--spec` or explicit payload/cap sequences. Do not add `--repeat-count`.
- If the user asked to open outputs in Finder, do not stop after `compile`;
  after `materialize`, open the root and verify `manifest/indexes/sequence_index.tsv`,
  per-design `sequences/forward.gb`, `plots/secondary_structure.native.png`,
  `composition_overview.svg`, `composition_overview.png`, and a
  secondary-structure subtitle containing the mismatch profile.
- If sequence subcomponents are missing, report the exact missing IDs or the
  primitive route needed; cap IDs require explicit 5'->3' sequence/source;
  do not present catalog JSONs as the requested deliverables.
- If `S0` is non-ligatable and the user explicitly says it is a control, rerun with the S0-control opt-in and verify `scar_nick.s0_match_required=false`.
- For missing constraints, name the missing fields and the primitive route.
- For generated artifacts, name the output directory and contracts produced.

4. Pair when the work widens.
- Pair with `harness-engineering` for skill routing, deterministic checks, or
  agent-execution reliability.
- Pair with `code-change-discipline` for contract, ontology, fail-fast, or
  module-boundary changes.

## Guardrails

- IDs select and validate provided parts; catalogs freeze references.
- Snapback and scar-nick solve primitives; the compiler emits one selected MSD
  unit per design.
- Study code must not import `dnadesign.cruncher.src.*` or parse Cruncher
  workspace internals; use `dnadesign.cruncher.snapback` and
  `dnadesign.cruncher.scar_nick` public primitive-export APIs.
- Rank ranges, rank lists, and all-hit selectors are valid source language only
  when a future expansion contract is explicit; current product compilation
  must fail fast instead of running implicit combinatorics.
- Scar-nick source refs project only four-base left/right basal spans into the
  final ssDNA unless a future contract selects more.
- Construct owns generic sequence assembly, not Retron biology.
- Folding has no workspace; it consumes producer bundles or explicit files.
- BaseRender renders visual contracts; it does not run ViennaRNA.
- Reader consumes frozen design catalogs, not live dnadesign workspaces.
- Do not say "snapshot posture" or lead with current phase unless the user
  asked for progress/status.

## Required Deliverables

- Input completeness classification.
- Selected route: compile, Snapback, scar-nick, YIU contrast, or status.
- Exact command or next file to open.
- Output directory/contract posture.
- Deliverable verification for materialize requests: record count, bundle root,
  GenBank/native-structure-PNG/review-SVG/review-PNG counts, or exact blockers.
- Fail-fast checks that apply.
- Primitive source selector posture when a spec references solver outputs.
- Residual unknowns or handoff route.

## Output

Return a short routing answer with:
- what parts are present and missing
- what will run next
- where outputs should live
- whether artifacts were emitted or which sequence subcomponents blocked them
- which invariant protects against drift
- status/preflight details only when explicitly requested

## Trigger Tests

Should trigger:
- "Compile this Retron MSD shorthand ID into a design catalog."
- "Generate one MSD sequence with GenBank and PNG outputs for these Retron IDs."
- "Open a transient Finder window with these Retron MSD outputs."
- "I have left/right bases but need to know if the scar-nick profile is valid."
- "Which primitive route owns this missing Retron MSD part?"
- "Harden the Retron MSD compiler skill or routing."

Should not trigger:
- "Run a generic Cruncher snapback search."
- "Explain retron biology broadly."
- "Design a wet-lab retron protocol."
- "Expose Retron MSD assembly as a generic top-level CLI."

## References

- [msd-design-references.md](references/msd-design-references.md), [route-matrix.md](references/route-matrix.md), [study-surfaces.md](references/study-surfaces.md), [refresh-loop.md](references/refresh-loop.md), [test-matrix.md](references/test-matrix.md), [external-sources.md](references/external-sources.md)
