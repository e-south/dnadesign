## OPS mental model

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-27

OPS owns neutral command routing, read-only observation, and deterministic
runbook control. Tool and study status runtime policy stays outside OPS core.

### What OPS is

OPS is the neutral orchestration and status shell for cross-tool work in `dnadesign`.
It does not own DenseGen, Infer, Notify, OPAL, USR, or study status runtime policy.
Repo-local scheduler assumptions must stay explicit: generic `ops runbook init` requires `--project <project>` or an explicit preset such as `--preset bu-scc-dunlop`; it does not silently infer site identity.
It owns:

- CLI dispatch
- catalog and status registry loading
- path and runbook contract handling
- generic state models and campaign aggregation
- generic study preflight execution
- runbook plan and execute services
- audit rendering and observation

### Plane model

| Plane | Public surface | What it does |
| --- | --- | --- |
| Discovery plane | `ops catalog` | Browse registered procedures, ownership docs, and related routes. |
| Observation plane | `ops progress` | Read one status surface or one manifest without mutating state. |
| Control plane | `ops runbook` | Initialize, plan, inspect active jobs, and execute orchestration. |
| Record plane | observed by snapshot surfaces | Checked-in study records, manifests, dataset contracts, and audit files already on disk. |
| Execution-readiness plane | observed by preflight surfaces | Host, workspace, scheduler, and command blockers for the next action. |

The plane a command belongs to is not always the same as the plane it observes.
Example: `ops progress show studies.stress-ethanol-cipro-growth.preflight` belongs to the observation plane, but it observes the execution-readiness plane.
When a route or workflow needs more nuance than the plane enum provides, keep
`Plane` on the canonical enum and add a separate field such as `Surface role`
instead of inventing replacement plane names.

### State semantics

OPS uses one global state lattice:

- `ok`: required evidence exists and satisfies the contract
- `attention`: evidence exists, but it shows an unsatisfactory or action-needed posture
- `missing`: required evidence or artifact is absent or unreadable

For study records, `ok` means the checked-in posture is coherent for the
declared current item. That item may be a phase in a sequential study or a track
in an open-ended study. Planned future outputs and historical upstream targets
can remain in evidence without escalating the current item to `attention`.

Severity order is global, not subsystem-local:

`missing > attention > ok`

That same precedence is used for campaign summaries, preflight blocker ordering, and other generic aggregation.

### Snapshot versus preflight

Use the cheap snapshot first:

- `uv run ops progress show studies.stress-ethanol-cipro-growth.status --json`

Snapshot answers record-backed questions:

- which study is active
- which datasets and row counts are checked in
- which surfaces are declared
- which phase, track, or route is current according to the record

Escalate to preflight when the question is about blockers or readiness:

- `uv run ops progress show studies.stress-ethanol-cipro-growth.preflight --scope next --command-timeout-seconds 30 --json`

Preflight answers readiness questions:

- which command fails now
- whether required paths exist
- whether environment flags are set
- whether GPUs or scheduler posture are sufficient
- whether a runbook plan compiles cleanly

For runbook orchestration, keep runtime visibility explicit:

- `scheduler_probe_state` answers whether OPS could query scheduler posture (`ok`, `skipped`, `unavailable`, `unsupported`, `error`)
- `active_job_resolution_state` answers whether active-job posture is known (`no_match`, `matched`, `multiple_matches`, `unknown`, `not_required`)
- `ops runbook plan` may still be useful when runtime visibility is degraded because it records `runtime_visibility` and blocks submit posture explicitly instead of guessing
- `ops runbook execute --submit` fails closed by default when active-job posture is unknown; `--allow-unknown-active-jobs` is the explicit degraded-mode override

### Escalation path

1. Discover with `uv run ops catalog list --simple`
2. Inspect one route with `uv run ops catalog show <registry-id>`
3. Read required inputs with `uv run ops progress explain <registry-id>`
4. Read cheap repo-backed posture with `uv run ops progress show ...`
5. Escalate to preflight when blockers matter
6. Plan and execute with `uv run ops runbook ...`
7. Read audit JSON back through `uv run ops progress show ops.control-plane.orchestration --audit-json ...`

### Source of truth

| Need | Source of truth |
| --- | --- |
| registered procedures and route metadata | `ops catalog` plus checked-in `*.registry.yaml` files |
| status-kind ontology | checked-in `*/ops/status.registry.yaml` and `ops/providers/*/status.registry.yaml` files |
| study lifecycle order, track map, and declared preflight checks | checked-in `docs/studies/<study-id>/operations/ops.study.yaml` |
| live study summary | `stress-ethanol-cipro-growth-status` snapshot plus the checked-in study record |
| execution blockers | `stress-ethanol-cipro-growth-preflight` |
| orchestration outcome | workspace-scoped audit JSON observed through `ops-audit-json` |
