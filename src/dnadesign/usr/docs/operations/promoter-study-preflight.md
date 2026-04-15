## Promoter Study Preflight

**Type:** contract
**Plane:** data-plane
**Owner-boundary:** usr
**Entry artifact:** one checked-in promoter-study directory plus study-owned execution surfaces
**Exit artifact:** one read-only command-level preflight summary for the active study
**Registry-id:** usr.data-plane.promoter-study-preflight
**Summary:** Run the active promoter-study preflight suite across DenseGen, Construct, Infer, Notify, and batch-plan contracts without mutating data or submitting jobs.
**Execution-kind:** iterative
**Status-kind:** promoter-study-preflight

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-13

Use this after the cheaper
[Promoter Study Status Contract](promoter-study-status-contract.md) when you
need command-level answers about what is ready, what is blocked, and why.
This route stays in the observation plane: it composes read-only preflight
checks from the study record and still leaves submit/execute work to
control-plane `ops runbook` commands.

### Choose the next surface

Use preflight after status answers the record question.

| Need | Surface | Why |
| --- | --- | --- |
| Where is the study now? | [Promoter Study Status Contract](promoter-study-status-contract.md) | Cheap record-plane snapshot of the checked-in study. |
| Which blockers remain on this host? | `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --json` | Command-level readiness for the next actionable phase. |
| Which owner surface should I open after blockers are clear? | `docs/studies/<study-id>/routes.md` | Study-owned one-hop handoff into DenseGen, Construct, Infer, LatentDNA, Cluster, or OPAL. |

This page is the blocker surface, not the downstream route map.

### First-thread bootstrap

Use this order when a thread starts cold:

1. Run [Promoter Study Status Contract](promoter-study-status-contract.md)
   first.
2. Run `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --json`
   only when the question is blocker or next-run readiness.
3. Open `docs/studies/<study-id>/routes.md` after blockers are clear and the
   next owner surface is the real need.

Fastest active-study preflight:

```bash
# Read the active study's command-level preflight summary as JSON.
uv run ops progress show usr.data-plane.promoter-study-preflight --json
```

Fastest next-phase preflight:

```bash
# Focus on the next actionable study phase.
uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --json
```

`--scope next` is the narrowest supported blocker gate. It is not guaranteed to
be cheap. If the current checked-in phase is a broad preparation phase such as
`infer_batch_preparation`, OPS will still run every declared Infer, Notify, and
runbook-plan check attached to that phase. Treat that cost as part of the
checked-in phase contract, not as a reason to bypass the blocker surface.

If the checked-in study contract marks notify-enabled Infer presets as
blocking, missing webhook env, missing TLS env, failed `notify profile doctor`,
failed `notify setup resolve-events`, or failed notify-enabled
`ops runbook plan` checks stay blocking rather than advisory. If the checked-in
study contract pins a GPU-only lane, report the exact selector from the study
record or returned preflight metadata instead of hard-coding that lane here.

If you need to pin a non-active study or you are invoking the command from
outside the repo checkout, add:

```bash
# Pin a specific checked-in study directory and emit the same preflight summary.
uv run ops progress show usr.data-plane.promoter-study-preflight \
  --repo-root <repo-root> \
  --study-dir docs/studies/<study-id> \
  --json
```

This route is read-only. It does not submit jobs, mutate USR datasets, or
advance Notify cursors. It composes explicit command preflights from the study
record and uses `ops.study.yaml` to decide which phases belong to the next
actionable scope versus the full study surface:

- `path_exists` and `dataset_snapshot` checks over declared study artifacts
- study-owned freshness checks when merged anchor or Construct context
  datasets trail upstream row counts despite still being materialized
- `workspace_layout` checks over declared Construct and Infer workspaces
- `environment` checks for webhook and TLS contracts
- `gpu_availability` checks for local infer posture when that scope is relevant
- `command` checks for DenseGen config probing, Construct doctor/runtime
  validation, Infer config validation, Infer dry-runs, Notify profile doctor,
  and Notify event-path resolution
- `scheduler_queue` checks for declared submit-threshold posture
- `runbook_plan` checks for DenseGen and Infer batch presets

`ops.study.yaml` is the visible source of readiness shape: it declares phases,
groups, artifacts, execution surfaces, and generic checks. The promoter-family
adapter still normalizes family-local paths and derived refs, but it no longer
hides a second imperative readiness graph behind the contract.

### Contract rules

- Start with `docs/studies/index.yaml` or pass `--study-dir`
  explicitly. Do not scan the repo for a best guess.
- Resolve relative study paths against the repo root, not the shell cwd.
- Fail visibly when `campaign.yaml`, `datasets.yaml`, `status.md`,
  `ops.study.yaml`, or declared execution surfaces are missing.
- Keep degraded state explicit:
  - missing datasets => `missing`
  - failed command preflights => `attention`
  - stale downstream handoffs => `attention`
  - blocked GPU-only lanes remain visible; there is no hidden 20B -> 7B fallback
- Use `ops.study.yaml` as the OPS-facing source of lifecycle phase order,
  execution surfaces, snapshot scope, and preflight phase-target grouping.
- Use `ops.study.yaml` as the OPS-facing source of declared artifacts,
  execution surfaces, and preflight scope/check planning.
- Use `pipeline.yaml`, when present, only as supplemental study-owned runtime
  context for exact Construct, Infer, or Notify mappings that are not worth
  duplicating in generic docs.
- Derive Infer Notify profile paths from the checked-in Infer lane configs
  rather than duplicating those profile paths in `pipeline.yaml`.
- Read blocker metadata from the returned ontology fields:
  `observes_plane`, `summary_scope`, `scope`, `phase_id`, `check_group`,
  `kind`, `surface_id`, and `artifact_id`.

### Minimum blocker evidence

| Question | Command | Minimum evidence | Fail visibly when |
| --- | --- | --- | --- |
| Which blockers remain on this host? | `usr.data-plane.promoter-study-preflight --scope next --json` | `scope`, `phase_id`, `check_group`, `kind`, `surface_id`, `artifact_id`, `state` | required study files or declared execution surfaces are missing |
| Which environment contract is broken? | same surface | `kind=environment` or `kind=command` plus the failing surface or artifact id | the study record omits the declared environment dependency |
| Which owner surface should I open after blockers clear? | `docs/studies/<study-id>/routes.md` | owner tool, primary doc or workspace, first command | blockers are clear but no study-owned route map exists |

### What this route is for

- status refresh before deciding the next command
- read-only readiness checks on login or CPU-only nodes
- explicit blocker reporting for missing USR datasets, Notify secret contracts,
  TLS/profile contracts, solver backends, and GPU-only lanes

### What this route is not for

- replacing tool-local runbooks
- mutating datasets or materializing new study artifacts
- hiding missing prerequisites behind fallback behavior

### Typical use

1. Run `usr.data-plane.promoter-study-status` for the cheap study snapshot.
   That summary is repo-scoped, observes the record plane, and does not
   elevate solely because the local host lacks a GPU.
2. Run `usr.data-plane.promoter-study-preflight` when you need command-level
   blockers before the next DenseGen, Construct, Infer, or Notify step. Use
   `--scope next` when you want the immediate execution-readiness blockers for
   the next actionable phase rather than the full historical surface. For
   broad preparation phases, that still means every declared check attached to
   that phase may execute before the summary is returned.
   The returned `checks` list is generic and traceable back to the checked-in
   study contract through `kind`, `surface_id`, and `artifact_id`.
3. Use the returned `checks` list to decide whether the next concrete action is:
   - grow DenseGen again
   - materialize the merged anchor set
   - materialize Construct contexts
   - fix Notify secret/profile contracts
   - move Infer execution to a GPU node that satisfies the checked-in lane
     contract reported by the study record or returned preflight metadata
