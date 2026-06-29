## OPS runtime visibility

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-29

Scheduler probes, active-job resolution, and degraded submit behavior in
`ops runbook` follow this contract.

### Why this exists

OPS keeps scheduler visibility explicit because active-job posture affects
duplicate-submit protection and `-hold_jid` wiring. Unknown scheduler posture is
not the same thing as `no active jobs`.

### Scheduler probe states

| State | Meaning |
| --- | --- |
| `ok` | scheduler probe returned usable evidence |
| `skipped` | no scheduler probe was attempted |
| `unavailable` | scheduler client or top-level probe was unavailable |
| `unsupported` | scheduler surfaced a candidate OPS job but not the explicit identity tags required for safe matching |
| `error` | scheduler probe returned malformed or incomplete detail output |

### Active-job resolution states

| State | Meaning |
| --- | --- |
| `not_required` | active-job posture was not required for the current path |
| `no_match` | scheduler evidence was available and no matching OPS jobs were found |
| `matched` | one matching OPS job was found |
| `multiple_matches` | more than one matching OPS job was found |
| `unknown` | OPS could not derive active-job posture safely |

### Command semantics

- `ops runbook active-jobs` fails fast when scheduler visibility is unavailable
  or active-job posture is `unknown`.
- `ops runbook plan` may still emit a usable plan when runtime visibility is
  degraded, but the JSON must include `runtime_visibility`, `warnings`, and a
  blocked submit posture when active-job posture is unknown.
- `ops runbook execute --submit` fails closed by default when
  `runtime_visibility.active_job_resolution_state=unknown`.
- `ops runbook execute --submit --allow-unknown-active-jobs` is the explicit
  degraded-mode override. Audit JSON records the override in `plan`.

### Explicit identity contract

OPS-submitted jobs carry an explicit scheduler identity contract:

- operator-visible job name derived from the runbook id and run-group digest
- machine-readable scheduler tags for `ops_run_group_id`,
  `ops_workspace_id`, and `ops_workflow_id`

Discovery matches the explicit OPS identity contract. It does not fall back to
workspace-path token guessing.

### Manual overrides

- `--active-job-id` remains the safe manual fallback when operators already know
  the job ids that should be chained or blocked against.
- `--no-discover-active-jobs` with no manual ids leaves active-job posture
  unknown, so submit is blocked by default.

### Supported diagnostics

```bash
uv run ops runbook diagnostics session-counts --qstat-file <fixture>
uv run ops runbook diagnostics submit-shape-advisor --qstat-file <fixture> --planned-submits <N> --warn-over-running 3
uv run ops runbook diagnostics operator-brief --qstat-file <fixture> --planned-submits <N> --warn-over-running 3
```
