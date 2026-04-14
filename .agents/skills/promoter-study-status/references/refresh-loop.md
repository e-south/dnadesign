# Refresh Loop

Use the checked-in study record first, then refresh the smallest explicit status
surface that answers the question.

## Required record inputs

- `docs/studies/README.md`
- `docs/studies/index.yaml`
- `docs/studies/<study-id>/campaign.yaml`
- `docs/studies/<study-id>/datasets.yaml`
- `docs/studies/<study-id>/status.md`
- `docs/studies/<study-id>/ops.study.yaml`
- `docs/studies/<study-id>/routes.md` when present
- `docs/studies/<study-id>/pipeline.yaml` when present

## Snapshot-first refresh

- `uv run ops progress show usr.data-plane.promoter-study-status --json`
- Use this for current phase, current datasets, row counts, downstream posture,
  and the next declared study surface.

## Explicit escalation for blockers

- `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --json`
- Use this when the question is `what blocks execution here?` or `what should
  run next on this host?`
- For the active `stress_ethanol_cipro_growth` study, the default notify-enabled Infer presets remain the strict submit-readiness contract.

## Record refresh helpers

- `uv run ops progress campaign --repo-root <repo-root> --manifest docs/studies/<study-id>/campaign.yaml`
- Use this when the checked-in campaign manifest needs a fresh summary across
  tracked procedures.

## Affiliated dataset sync posture

- Keep sync posture in `datasets.yaml`, not in the top-level skill.
- Use `usr.data-plane.hpc-sync` when a sync-enabled dataset needs explicit audit
  evidence.
- Preserve `onboard_mode: existing_remote` plus `strict_bootstrap_id: true`
  when the first local pull must bind to an explicit remote dataset id.

## Source and handoff language

- Use `source/handoff mode` when the canonical consolidated feature dataset is
  still planned.
- Do not invent feature-matrix or downstream campaign readiness when the
  checked-in study record does not declare it.
