## Workflow Router

Use this router before emitting any scheduler commands.

### Route keys

- `workflow_id`: one path from the list below
- `execution_locus`: `local_shell`, `scc_login_shell`, `ondemand_shell`, `ondemand_app_shell`, or `unknown`
- `session_handoff_state`: `none`, `session_request_pending`, or `session_ready`

### DenseGen notify policy

- For runbook-native Ops execution, DenseGen uses notify by default and routes to `densegen_batch_with_notify_slack`.
- Use `densegen_batch_submit` only when the request explicitly opts out (`--no-notify`, "without notify", "notify off").

### Routes

#### `densegen_batch_submit`

Trigger cues:
- "start a densegen workspace ... batch job ... without notify"
- "run densegen ... notify off"
- "use --no-notify"

Execution contract:
1. Load `probe-first-contract.md`, `session-status-reporting.md`, `submission-shape-advisor.md`, and `batch-submit-contract.md`.
2. Validate DenseGen config path and solver probe.
3. Validate output placement and project quota context.
4. Run submit QA preflight on selected template.
5. Run session-status check and apply warning when running_jobs > 3.
6. Run shape advisor for planned submit count.
7. Render operator brief for user-facing submit gate.
8. Submit `docs/bu-scc/jobs/densegen-cpu.qsub`.
9. Capture job id and log path.

Required evidence:
- config validation output
- storage and quota check output
- qa preflight pass output
- session-status output
- shape-advisor output
- operator-brief output
- densegen job handle

#### `densegen_batch_with_notify_slack`

Trigger cues:
- "start a densegen workspace ... batch job ..."
- "start densegen ... and wire up notify/slack"
- "run a densegen workflow stress ethanol and cipro workspace for two hours"

Execution contract:
1. Scaffold or select runbook via Ops (`uv run ops runbook precedents` or `uv run ops runbook init --workflow densegen ...`).
2. Set up notify profile via `uv run notify setup slack ...`.
3. Run submit QA preflight for watcher and densegen templates.
4. Run session-status check and apply warning when running_jobs > 3.
5. Run shape advisor for planned submit count.
6. Render operator brief for user-facing submit gate.
7. Submit watcher job with `NOTIFY_PROFILE`.
8. Submit DenseGen batch job.
9. Capture both job ids and log paths.
10. Monitor both jobs with bounded loops.

Required evidence:
- notify profile path
- qa preflight pass output for both templates
- session-status output
- shape-advisor output
- operator-brief output
- watcher job handle
- densegen job handle

Boundary:
- Notify consumes USR `.events.log`, not DenseGen `outputs/meta/events.jsonl`.

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
- other SGE operations without DenseGen/Notify/OnDemand-specific language
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
