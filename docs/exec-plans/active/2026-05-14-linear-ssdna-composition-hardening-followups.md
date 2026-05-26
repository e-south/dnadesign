## Exec plan: Linear ssDNA composition hardening follow-ups

**Status:** active
**Owner:** Shockwing / Codex handoff
**Created:** 2026-05-14
**Last updated:** 2026-05-14
**Authority:** active follow-up tracker; not the generic operator route

### Purpose / Big Picture

This plan tracks the remaining hardening work after the local Construct,
Folding, BaseRender, and Retron dogfood path landed through the two-row
composition review. The completed implementation record lives at
[2026-05-13 generic linear ssDNA composition](../completed/2026-05-13-generic-linear-ssdna-composition.md).

The goal is to keep the architecture decoupled while avoiding workspace and
file sprawl: Construct composes declared sequence products, Folding owns
ViennaRNA execution and native plot publication, BaseRender renders linear
visual contracts, and study/Reader-facing design references stay frozen
handoff records.

### Progress

- [x] (2026-05-14 00:00Z) Split completed implementation history out of the
  active plan and kept this follow-up plan focused on remaining hardening.

### Surprises & Discoveries

The first implementation plan became useful historical evidence but too noisy
as an active router. Keeping it under `completed/` prevents phase-ghost
answers while preserving the implementation audit trail.

### Decision Log

- Keep Folding as a workspace-less service boundary for ViennaRNA preflight,
  execution, parsing, and plotting.
- Keep Retron MSD label compilation under study-owned helpers and the repo-local
  skill; do not expose a top-level `retron-msd` tool.
- Do not add repeat-expanded visual/folding evidence unless a future contract
  explicitly introduces that surface.

### Outcomes & Retrospective

Pending. Close this plan when the remaining follow-ups are either implemented
or deliberately split into smaller accepted plans.

### Context and Orientation

Read these first:

- [Generic linear ssDNA composition spec](../../dev/plans/cross-tool/linear-ssdna-composition/2026-05-13-generic-linear-ssdna-composition.md)
- [Completed implementation record](../completed/2026-05-13-generic-linear-ssdna-composition.md)
- [Retron linear ssDNA handoff](../../studies/retron_hairpin_design/contexts/composition/linear-ssdna-composition.md)
- [ADR 0002](../../architecture/decisions/adr-0002-generic-linear-ssdna-composition.md)

### Plan of Work

1. Evaluate optional USR persistence only after local artifact contracts remain
   stable across real selected designs.
2. Add source-ref dogfood when fresh `de033` and `scar_nick_teto` outputs can
   support the claims without hand-authored shortcuts.
3. Split large Construct/Folding modules before adding more visual/export
   behavior.
4. Keep Retron MSD design-reference outputs transient unless an owning Reader
   experiment snapshots a catalog and assets under its own `inputs/designs/`.

### Concrete Steps

For module splitting, preserve behavior and tests:

- Split `src/dnadesign/folding/src/viennarna_svg.py` into DOM parsing,
  orientation/geometry, label placement, annotation, and canvas helpers.
- Split `src/dnadesign/folding/src/api.py` into request loading, preflight, and
  backend execution modules.
- Split `src/dnadesign/construct/src/composition.py` into assembly engine,
  bundle writer, export writers, and render/folding handoff modules.

For source-ref dogfood, require real producer artifacts:

- Use fresh `de033` released-product outputs for cap/snapback inputs.
- Use fresh `scar_nick_teto` outputs for left/right base-junction inputs.
- Fail fast when a source ref cannot prove sequence identity, orientation, or
  coordinate frame.

### Validation and Acceptance

- `uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .`
- `uv run python -m dnadesign.devtools.docs.checks`
- `uv run pytest -q src/dnadesign/contracts/tests src/dnadesign/folding/tests src/dnadesign/construct/tests src/dnadesign/studies`
- Targeted CLI smoke tests for `construct compose`, `folding preflight/run/plot`,
  and the Retron MSD compiler.
