# Cruncher Study Preflight

**Type:** contract
**Plane:** data-plane
**Owner-boundary:** cruncher
**Entry artifact:** one checked-in cruncher-study directory plus declared workspace and validation-command surfaces
**Exit artifact:** one read-only command-level preflight summary for the selected cruncher study
**Registry-id:** cruncher.data-plane.cruncher-study-preflight
**Summary:** Run the current cruncher-study preflight suite across declared workspace-layout and command-validation surfaces without mutating outputs.
**Execution-kind:** iterative
**Status-kind:** cruncher-study-preflight

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-21

Use this route after the cheaper study snapshot when you need read-only
command readiness for the current Cruncher study phase.

## Quick route

- Read-only preflight:
  `uv run ops progress show cruncher.data-plane.cruncher-study-preflight --study-dir docs/studies/<study-id> --scope next --json`
- Snapshot first:
  `uv run ops progress show cruncher.data-plane.cruncher-study-status --study-dir docs/studies/<study-id> --json`

## What this route checks

- study-owned record files such as `routes.md` and `pipeline.yaml`
- declared workspace roots
- declared read-only validation commands such as `released-target-search` and `yiu validate`

## Notes

- `next` scope focuses the current actionable phase and its shared study-record checks.
- `full` scope runs the whole declared suite.
- This route does not submit jobs or write Cruncher outputs.
