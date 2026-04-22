## Snapback Shortening Effort Routes

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-21

Use this page after the tracked study status answers `where are we?`.
Use preflight when you need blocker or command-readiness answers.
This page keeps the study-owned handoff map in one place.

### Quick route

- Snapshot:
  `uv run ops progress show cruncher.data-plane.cruncher-study-status --study-dir docs/studies/snapback_shortening_effort --json`
- Preflight:
  `uv run ops progress show cruncher.data-plane.cruncher-study-preflight --study-dir docs/studies/snapback_shortening_effort --scope next --json`
- Repo-local study shortcut:
  `.agents/skills/snapback-hairpin-study/SKILL.md`
- Pair with:
  `harness-engineering` for study-surface hardening and
  `pragmatic-programming-principles` for boundary or contract changes.

### Boundary shorthand

- `released-product Snapback` means the dual-enzyme precursor lane where final geometry is evaluated on the retained post-release product.
- `preserved-site Snapback` means the older one-enzyme lane and stays a separate contract.
- `YIU` means mismatch-centric payload rendering over a fixed 4 nt internal window; it is not the shortening topology engine here.
- `retron context` means biological framing from the checked-in audit notes, not scoring hooks or implicit solver relaxations.

### Primary route: released-product Snapback

Use this route when the task is actual shortening construction or evaluation.
This is the active study lane.

- Type: `route`
- Plane: `data-plane`
- Surface role: `primary-execution`
- Owner-boundary: `cruncher`
- Current state: `in_progress`
- Workspace: `src/dnadesign/cruncher/workspaces/demo_snapback`
- Primary doc:
  `src/dnadesign/cruncher/docs/guides/snapback_released_workflow.md`
- First read-only command:
  `uv run cruncher snapback released-target-search --workspace-root src/dnadesign/cruncher/workspaces/demo_snapback --nick-additional-path src/dnadesign/cruncher/workspaces/demo_snapback/inputs/nickases/local.nickases.yaml --release-additional-path src/dnadesign/cruncher/workspaces/demo_snapback/inputs/release_enzymes/local.release.yaml --nick-boundary 0 --paired-bp 3 --cap-nt 3 --json`
- Follow-up mutating commands:
  `cd src/dnadesign/cruncher/workspaces/demo_snapback && uv run cruncher snapback released-design --spec configs/snapback/demo_released_origin_033.released.snapback.yaml --force-overwrite`
  `cd src/dnadesign/cruncher/workspaces/demo_snapback && uv run cruncher snapback released-show --run outputs/released_design --json`
- Route note:
  use this route for the actual shortening construction model.

### Contrast route: YIU boundary check

Use this route only when you need a contrast surface for boundary language or a
reminder of what YIU does and does not model.

- Type: `route`
- Plane: `data-plane`
- Surface role: `contrast-check`
- Owner-boundary: `cruncher`
- Current state: `planned`
- Workspace: `src/dnadesign/cruncher/workspaces/demo_monotypic_tetr`
- Primary docs:
  `src/dnadesign/cruncher/docs/guides/yiu_workflow.md`
  `src/dnadesign/cruncher/workspaces/demo_monotypic_tetr/runbook.md`
- First read-only command:
  `cd src/dnadesign/cruncher/workspaces/demo_monotypic_tetr && uv run cruncher yiu validate --spec configs/yiu/tetr_teto2_wt_direct.yiu.yaml`
- Follow-up mutating commands:
  `cd src/dnadesign/cruncher/workspaces/demo_monotypic_tetr && uv run cruncher yiu render --spec configs/yiu/tetr_teto2_wt_direct.yiu.yaml --force-overwrite --emit-renders`
  `cd src/dnadesign/cruncher/workspaces/demo_monotypic_tetr && uv run cruncher yiu show --bundle outputs/plots/yiu__tetr_teto2_wt_direct`
- Route note:
  use this route only to keep the YIU boundary explicit and auditable.

### Context surfaces

- Study note:
  `docs/studies/snapback_shortening_effort/status.md`
- Study command ladder:
  `docs/studies/snapback_shortening_effort/pipeline.yaml`
- Consolidated retron/P4 and YIU executive summary:
  `src/dnadesign/cruncher/docs/dev/2026-04-19-retron-p4-hairpin-variant-audit.md`
- Route note:
  these notes are study context, not executable contracts.
