# Refresh Loop

Start with the checked-in study record, then refresh the smallest status
surface that answers the question.

## Blank-thread bootstrap

1. Read `docs/studies/README.md` and `docs/studies/index.yaml`.
2. Run `uv run ops progress show studies.stress-ethanol-cipro-growth.status --json`.
3. Escalate to
   `uv run ops progress show studies.stress-ethanol-cipro-growth.preflight --scope next --json`
   only for blocker or next-run readiness questions.
4. Open `docs/studies/<study-id>/routes.md` only after the record or blocker
   question is answered. Open `routes/opal.md` or `routes/latentdna.md` only
   after that owner surface is selected.

## Required record inputs

- `docs/studies/README.md`
- `docs/studies/index.yaml`
- `docs/studies/<study-id>/campaign.yaml`
- `docs/studies/<study-id>/datasets.yaml`
- `docs/studies/<study-id>/status.md`
- `docs/studies/<study-id>/ops.study.yaml`
- `docs/studies/<study-id>/routes.md` when present
- `docs/studies/<study-id>/routes/` detail files only after route selection
- `docs/studies/<study-id>/pipeline.yaml` when present

## Snapshot-first refresh

- `uv run ops progress show studies.stress-ethanol-cipro-growth.status --json`
- Use this for current phase, current datasets, row counts, downstream posture,
  and the next declared study surface.

## Explicit escalation for blockers

- `uv run ops progress show studies.stress-ethanol-cipro-growth.preflight --scope next --json`
- Use this for `what blocks execution here?` or `what should run next on this
  host?`

## Minimum evidence by question

| Question | Primary surface | Minimum evidence | Fail visibly when |
| --- | --- | --- | --- |
| Where is the study now? | `studies.stress-ethanol-cipro-growth.status` | study id, current phase, dataset ids, row counts, downstream posture, next surface | selector fields or required record files are missing |
| Which exploratory-analysis artifacts are available? | `studies.stress-ethanol-cipro-growth.status` plus `routes.md` | `analysis_surfaces.densegen`, `analysis_surfaces.latentdna`, or `analysis_surfaces.cluster` with ids, commands, and artifact paths | route inventory is missing for the owning tool or the study omits the needed workspace/doc binding |
| What blocks execution here? | `studies.stress-ethanol-cipro-growth.preflight --scope next` | `scope`, `phase_id`, `check_group`, `kind`, `surface_id`, `artifact_id` | `ops.study.yaml` or declared execution surfaces are missing |
| Which dataset sync posture is current? | `datasets.yaml` plus `usr.data-plane.hpc-sync` | dataset id, remote profile, audit JSON path, explicit drift summary | sync-enabled dataset entries or audit evidence are missing |
| Which owner surface should I open next? | `docs/studies/<study-id>/routes.md` | owner tool, entry artifact, primary doc or workspace, first command | the study spans owner surfaces but no route map is checked in |

## Record refresh helpers

- `uv run ops progress campaign --repo-root <repo-root> --manifest docs/studies/<study-id>/campaign.yaml`
- Use this when the checked-in campaign manifest needs a fresh summary.

## Affiliated dataset sync posture

- Keep sync posture in `datasets.yaml`, not in the top-level skill.
- Use `usr.data-plane.hpc-sync` when a sync-enabled dataset needs explicit audit
  evidence.
- Preserve `onboard_mode: existing_remote` plus `strict_bootstrap_id: true`
  when the first local pull must bind to an explicit remote dataset id.

## Failure routing

- Missing registry, stale selector, or missing study directory: repair
  `docs/studies/index.yaml` and the required record files, then rerun status.
- Missing or incomplete command-level evidence: rerun
  `studies.stress-ethanol-cipro-growth.preflight --scope next` and summarize the
  blocker with `kind`, `surface_id`, and `artifact_id`.
- Missing sync posture: refresh the explicit `usr diff --audit-json-out` audit
  named by `datasets.yaml`, then summarize it through `usr.data-plane.hpc-sync`.
- Missing downstream handoff or stale source growth: report the checked-in
  `source/handoff mode` instead of inventing a downstream-ready state.

## Source and handoff language

- Use `source/handoff mode` when the canonical consolidated feature dataset is
  still planned.
- Do not invent feature-matrix or downstream campaign readiness when the
  checked-in study record does not declare it.
- If `ops.study.yaml` declares default notify-enabled Infer presets as the
  submit-readiness contract, keep those environment, profile, and plan blockers
  explicit.
