# Runbook Entrypoints

Prefer command-first Ops entrypoints over path-discovery heuristics when the
environment already exposes the runbook interface.

## Core commands

- `uv run ops catalog list --simple`
- `uv run ops catalog show <registry-id>`
- `uv run ops runbook presets`
- `uv run ops runbook init --workflow densegen|infer ... --project <project>`
- `uv run ops runbook init --workflow densegen|infer ... --preset <preset>`
- `uv run ops runbook plan --runbook <runbook.yaml>`
- `uv run ops runbook active-jobs --runbook <runbook.yaml>`
- `uv run ops runbook execute --runbook <runbook.yaml> --audit-json <audit.json> --no-submit`
- `uv run ops runbook execute --runbook <runbook.yaml> --audit-json <audit.json> --submit`
- `uv run ops runbook diagnostics session-counts --qstat-file <fixture>`
- `uv run ops runbook diagnostics submit-shape-advisor --qstat-file <fixture> --planned-submits <N> --warn-over-running 3`
- `uv run ops runbook diagnostics operator-brief --qstat-file <fixture> --planned-submits <N> --warn-over-running 3`
- `uv run ops progress show ops.control-plane.orchestration --audit-json <audit.json>`

## Usage rules

- start with `--no-submit` as the default pressure-test path before real queue
  mutation
- DenseGen scaffolds notify by default; use `--no-notify` only for explicit
  batch-only requests
- presets are explicit site shortcuts, not hidden defaults; `--no-discover-active-jobs`
  without manual `--active-job-id` leaves active-job posture unknown
- keep runbook selection aligned with the `route_id` chosen in
  `workflow-router.md`
