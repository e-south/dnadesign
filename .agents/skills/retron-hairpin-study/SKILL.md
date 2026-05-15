---
name: retron-hairpin-study
description: "Compile or route Retron MSD genetic-compiler requests. Use for MSD IDs, single-unit MSD sequence bundles, design catalogs, GenBank/PNG, or missing MSD parts. Do not use for generic Cruncher/snapback or bench protocols."
metadata:
  version: 0.7.3
  category: workflow-automation
  tags: [retron, msd, genetic-compiler, snapback, scar-nick, composition, study]
---

# Retron Hairpin Study

## Purpose

Route Retron MSD work as a genetic compiler. The default job is not to report
study phase. It is to decide whether the user has provided enough parts to
compile a design reference or one MSD sequence unit, or whether missing
constraints must route to a primitive solver.

## Scope

In scope:
- Retron MSD shorthand IDs and explicit part sets.
- Study-owned `msd_design_reference_v1` / `msd_design_catalog_v1` compilation.
- Routing missing cap/shortening constraints to Snapback.
- Routing missing stem-base or terminal-nick constraints to scar-nick base-junction.
- Routing mismatch-display questions to YIU as contrast only.
- Construct/BaseRender service calls for one MSD unit per design after parts are
  selected.
- Skill, route-map, and compiler-harness hardening.

Out of scope:
- Generic Cruncher walkthroughs.
- Bench-level retron protocol advice.
- Making Retron MSD compilation a top-level `retron-msd` tool.
- Creating one Construct or Folding workspace per requested design.
- Reporting phase/status posture unless the user explicitly asks for study
  progress or blockers.

## Success Criteria

- The first decision is input completeness: compile now, or route missing
  constraints.
- Complete user-provided parts are validated and compiled without solver work.
- Incomplete parts route to the smallest primitive: Snapback, scar-nick, or
  YIU contrast.
- Sequence artifact output is one MSD unit per design: 5' flank + left base,
  payload primary, cap geometry, payload complement, right base + 3' flank.
- No user-facing repeat count; do not chain complete MSD units together.
- Outputs go to explicit transient or caller-owned directories; no workspace
  sprawl.
- Contracts fail fast on profile drift, non-ligatable `S0=M` violations, unknown registry
  parts, or missing artifacts.
- Status/preflight commands are optional progress tools, not default answer
  posture.

## Workflow

1. Classify the request.
- Complete MSD label or complete parts: use
  [msd-design-references.md](references/msd-design-references.md).
- Need sequence, visual, or GenBank from selected parts: compile a reference
  first, then materialize a single-unit sequence bundle.
- Missing cap, shortening, or stem/cap geometry: route to Snapback in
  `docs/studies/retron_hairpin_design/routes.md`.
- Missing left/right base feasibility, terminal-nick route, nickase, or
  `S3/S2/S1/S0` profile: route to scar-nick.
- Mismatch-only display or boundary contrast: route to YIU; it is not the
  topology engine.
- Progress or blocker question: only then use
  `cruncher-study-status` or `cruncher-study-preflight`.

2. Load only the needed surfaces.
- Compiler route: `docs/studies/retron_hairpin_design/routes.md`, then
  `docs/studies/retron_hairpin_design/msd_design_registry.yaml`, then
  `references/msd-design-references.md`.
- Whole-product context: `docs/studies/retron_hairpin_design/linear-ssdna-composition.md`,
  then the active exec plan.
- Machine-readable command groups: `docs/studies/retron_hairpin_design/pipeline.yaml`.
- Ownership boundaries: [study-surfaces.md](references/study-surfaces.md).

3. Execute or report the route.
- For complete reference inputs, run `uv run python -m dnadesign.studies.retron_hairpin_design.cli lint|compile`.
- For GenBank/PNG output, run the same module's `materialize` command with
  explicit payload/cap sequences. Do not add `--repeat-count`.
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
- Fail-fast checks that apply.
- Residual unknowns or handoff route.

## Output

Return a short routing answer with:
- what parts are present and missing
- what will run next
- where outputs should live
- which invariant protects against drift
- status/preflight details only when explicitly requested

## Trigger Tests

Should trigger:
- "Compile this Retron MSD shorthand ID into a design catalog."
- "Generate one MSD sequence with GenBank and PNG outputs for these Retron IDs."
- "I have left/right bases but need to know if the scar-nick profile is valid."
- "Which primitive route owns this missing Retron MSD part?"
- "Harden the Retron MSD compiler skill or routing."

Should not trigger:
- "Run a generic Cruncher snapback search."
- "Explain retron biology broadly."
- "Design a wet-lab retron protocol."
- "Expose Retron MSD assembly as a generic top-level CLI."

## References

- [msd-design-references.md](references/msd-design-references.md)
- [route-matrix.md](references/route-matrix.md)
- [study-surfaces.md](references/study-surfaces.md)
- [refresh-loop.md](references/refresh-loop.md)
- [origin-033-hits.md](references/origin-033-hits.md)
- [test-matrix.md](references/test-matrix.md)
- [external-sources.md](references/external-sources.md)
