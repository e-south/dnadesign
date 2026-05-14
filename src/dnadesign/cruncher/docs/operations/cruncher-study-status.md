# Cruncher Study Status Contract

**Type:** contract
**Plane:** data-plane
**Owner-boundary:** cruncher
**Entry artifact:** one checked-in cruncher-study directory with routes and pipeline context
**Exit artifact:** a read-only snapshot of the declared route or track, command groups, and operator context
**Registry-id:** cruncher.data-plane.cruncher-study-status
**Summary:** Read one checked-in cruncher-study directory and summarize its declared route or track, command groups, and native-agent bootstrap context.
**Execution-kind:** iterative
**Status-kind:** cruncher-study-status

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-21

Checked-in Cruncher study status reports the declared route or track, command
groups, operator context, and study-owned paths that explain what to open next.

## Quick route

- Current study snapshot:
  `uv run ops progress show cruncher.data-plane.cruncher-study-status --study-dir docs/studies/<study-id> --json`
- Command blockers for the same study:
  `uv run ops progress show cruncher.data-plane.cruncher-study-preflight --study-dir docs/studies/<study-id> --scope next --json`

## What this route reads

- `docs/studies/<study-id>/status.md`
- `docs/studies/<study-id>/routes.md`
- `docs/studies/<study-id>/pipeline.yaml`
- `docs/studies/<study-id>/ops.study.yaml`

## When to use it

- You are starting a new thread and need the narrowing context first.
- You want the maintained command groups instead of reconstructing them from workspaces.
- You need the explicit “do not collapse these abstractions together” list before running Cruncher.

## Notes

- If the active checked-in study belongs to a different family, pass `--study-dir` explicitly.
- This route is record-plane only. It does not mutate workspaces or run Cruncher commands.
