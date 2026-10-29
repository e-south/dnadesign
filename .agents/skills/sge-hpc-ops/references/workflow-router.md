## Workflow Router

Use this router before emitting any scheduler commands.

### Route keys

- `route_id`: one path from the list below
- `execution_locus`: `local_shell`, `scc_login_shell`, `ondemand_shell`, `ondemand_app_shell`, or `unknown`
- `session_handoff_state`: `none`, `session_request_pending`, or `session_ready`

### Routes

#### `infer_batch`

Trigger cues:
- "plan or submit an Infer batch"
- "run inference with Notify"

Execution contract:
1. Select the Infer configuration and its destination dataset.
2. Use `ops runbook init --workflow infer` with an explicit project or preset.
3. Runbooks enable notify by default; use `--no-notify` for an explicit batch-only request.
4. Check configuration, GPU resources, output storage, notification setup and active jobs using the runbook plan and probe-first contract.
5. Review `ops runbook execute --no-submit` output and scheduler verification before submission.
6. Retain the audit path, workload job ID and any watcher job ID; monitor with bounded checks.

Required evidence:
- runbook and Infer configuration paths
- resource, storage and active-job checks
- preflight results and planned commands
- audit path and submitted job IDs when execution was requested


#### `ondemand_session_request`

Trigger cues:
- "submit/request an interactive on demand session"

Execution contract:
1. Route to BU OnDemand documentation and portal.
2. Capture requested resource/time profile.
3. Ask user to confirm session appears in "My Interactive Sessions".
4. Return handoff checklist for in-session tasks.

Required evidence:
- OnDemand route taken
- session request confirmation checkpoint

#### `ondemand_session_handoff`

Trigger cues:
- "I have entered an OnDemand session ..."

Execution contract:
1. Set `session_handoff_state=session_ready`.
2. Skip session creation and run in-session probes (`hostname`, `pwd`, scheduler tool availability).
3. Run session-status check for current shell.
4. Continue requested task in current locus.
5. If task is long-running, route to batch submission and keep interactive shell for diagnostics.

Required evidence:
- execution locus probe output
- session-status output
- explicit handoff state in output

#### `generic_sge_ops`

Trigger cues:
- "help debug Eqw", "job tracking", "process reaper", "memory per core", "array conversion", "batch dependency pipeline"

Execution contract:
1. Use probe-first + mode-specific reference.
2. Run session-status report before submit/monitor decisions.
3. Run shape advisor before multi-submit proposals.
4. Render operator brief before new submission decisions.
5. For multi-job workflows, prefer arrays or explicit `-hold_jid` sequencing over unbounded parallel submits.
6. If more than 3 running jobs are present, warn and require explicit confirmation before additional submissions.
7. Respect queue fairness: no line-skipping or queue-bypass behavior.
8. Apply verify-before-submit, QA preflight checks, and bounded monitoring.
9. For reaper incidents, capture evidence before proposing retries.

Required evidence:
- capability snapshot
- session-status output
- shape-advisor output
- operator-brief output
- qa preflight result when submit templates are in scope
- run handle or diagnostic handle
