---
name: sge-hpc-ops
description: Operate SGE or UGE clusters including BU SCC with probe-first capability detection, deterministic interactive or batch execution, verify-before-submit gates, workflow routing for DenseGen plus Notify chains, and freshness checks for volatile SCC policy claims. Use when users ask about qsub, qrsh, qlogin, qstat, qdel, job arrays, resource requests, queue monitoring, BU SCC connectivity, OnDemand sessions, transfer-node usage, Notify Slack wiring, or BU SCC batch workflows. Do not use for non-SGE schedulers or local-only coding tasks.
metadata:
  version: 0.7.4
  category: workflow-automation
  tags: [hpc, sge, bu-scc, batch, operations]
---

# SGE HPC Ops

## Purpose

Run SGE-family workloads safely with explicit capability probing, workflow-aware routing, bounded execution loops, and submission guardrails that reduce retries and queue waste.

## Scope

In scope:
- scheduler capability probing before command generation
- workflow routing for DenseGen batch, DenseGen plus Notify Slack, OnDemand request, and OnDemand handoff
- deterministic interactive and batch execution planning
- verify-before-submit checks plus QA/runtime preflight gates
- active-job snapshot and session-status reporting for user-facing HPC status
- over-threshold submit guardrails when more than 3 jobs are already running
- submission-shape advisor guidance (`array` or `-hold_jid`) for multi-submit pressure
- queue fairness policy: respect the queue and do not skip the line
- BU SCC source freshness handling for volatile policy claims
- bounded monitor/recovery loops with process-reaper-aware triage

Out of scope:
- Slurm, PBS, or Kubernetes-only scheduling tasks
- local-only development tasks that do not touch a scheduler
- cluster administration changes (queues, host groups, complexes, projects)

## Input Contract

Minimum inputs:
- scheduler context (site, login host, expected scheduler family)
- workload intent (interactive debug, batch production, watcher, transfer)
- resource goals (runtime, cores, memory, GPU, project/account)
- target script/config paths and success conditions
- session context when available (`execution_locus`, `session_handoff_state`, prior job ids)
- submit intent when available (single submit, multi submit, dependency chain)

Clarification policy:
- ask at most two questions when scheduler identity or risk boundaries are unclear
- proceed with explicit assumptions when risk boundaries are clear
- log assumptions in final output

## Success Criteria

- exactly one route is emitted (`workflow_id`) before command generation
- capability snapshot includes scheduler facts, `execution_locus`, session status, and active-job snapshot
- user-facing status card is emitted before additional submissions
- verify-before-submit and qa preflight checks pass before real `qsub`
- if running job count is more than 3 and user asks for more submits, warning + explicit confirmation gate is enforced
- submission-shape advisor emits `array` or `hold_jid` guidance for over-threshold fanout
- execution plan respects queue fairness and avoids line-skipping behavior
- BU SCC volatile claims are refreshed when older than 45 days or marked unknown
- output bundle is reproducible across repeated runs

## Workflow

### Step 1: Route workflow and execution locus

- classify request using `references/workflow-router.md`
- select exactly one `workflow_id`:
  - `densegen_batch_submit`
  - `densegen_batch_with_notify_slack`
  - `ondemand_session_request`
  - `ondemand_session_handoff`
  - `generic_sge_ops`
- detect `execution_locus` (local shell, SCC login shell, OnDemand shell, OnDemand app shell, unknown)
- if user is already in OnDemand, route to `ondemand_session_handoff` and skip session-creation flow

### Step 2: Load minimum reference set (progressive disclosure)

- load base references first: `references/probe-first-contract.md` and `references/workflow-router.md`
- then load only the route-specific pack defined in `references/route-load-matrix.md`
- when reporting HPC status to users, include `references/session-status-reporting.md` and `references/user-status-contract.md`
- when submit-shape or readiness is in scope, include `references/submission-shape-advisor.md` and `references/operator-brief.md`
- when command-first Ops runbooks are available, include `references/runbook-entrypoints.md`
- for batch/interactive specifics, load only the matching contract (`references/batch-submit-contract.md` or `references/interactive-contract.md`)

### Step 3: Apply up-to-date handling

- for BU SCC claims, use official BU pages listed in `references/bu-scc-system-usage.md`
- classify claims as stable vs volatile (numeric limits, hostnames, quotas, process-reaper thresholds)
- check retrieval age in `references/source-evidence.md`; refresh volatile claims older than 45 days
- if volatile claims cannot be refreshed in-session, mark them unknown and avoid hard-coded values

### Step 4: Build a capability snapshot

- run probes from `references/probe-first-contract.md`
- run `scripts/sge-session-status.sh --warn-over-running 3`
- run `scripts/sge-active-jobs.sh --max-jobs 12` when status reporting is requested
- include: `workflow_id`, `execution_locus`, `session_handoff_state`, unresolved unknowns
- include counts: `running_jobs`, `queued_jobs`, `eqw_jobs` and threshold warning state

### Step 5: Build deterministic execution plan

- map workload to resource profile (CPU, GPU, watcher, transfer)
- align threaded workloads (`OMP_NUM_THREADS`) with `NSLOTS`
- prefer arrays (`SGE_TASK_ID`) for fanout and `-hold_jid` chains for ordered pipelines
- run `scripts/sge-submit-shape-advisor.sh --warn-over-running 3` before multi-submit plans
- if running jobs are more than 3 and additional submissions are requested, warn and require explicit confirmation
- in over-threshold cases, propose arrays or dependency chains before raw parallel submits
- respect the queue and do not skip the line
- for DenseGen + notify chains, follow `references/workload-dnadesign.md`

### Step 5a: Use runbook-native orchestration commands when available

- prefer command-first Ops entrypoints from `references/runbook-entrypoints.md`,
  including `uv run ops runbook precedents`, over path-discovery heuristics
- DenseGen defaults to notify-enabled runbooks; use `--no-notify` only for explicit batch-only requests, and treat `--no-submit` as the default pressure-test path before any real submit

### Step 6: Apply verify-before-submit gate

- verify project/account flags, script readability, and shell init compatibility (`#!/bin/bash -l`)
- run scheduler syntax validation (`qsub -verify`, fallback `qsub -w v`)
- run `scripts/qa-sge-submit-preflight.sh` on submit artifacts
- run:
  - `scripts/sge-session-status.sh --warn-over-running 3`
  - `scripts/sge-submit-shape-advisor.sh --warn-over-running 3`
  - `scripts/sge-operator-brief.sh --planned-submits <N> --warn-over-running 3`
- for notify workflows, verify resolver/profile inputs and webhook environment before watcher submit
- submit only when checks are green

### Step 7: Monitor and recover

- use bounded queue/log monitoring loops with timeout and teardown
- keep separate run handles for each submitted job (`densegen_job_id`, `notify_job_id`)
- track transitions with `qstat` and post-run evidence with `qacct` when available
- triage `Eqw` and process-reaper events before retries or mutations
- allow at most one guarded retry with explicit reason and user confirmation

### Step 8: Report using output contract

- report findings, interpretation, action commands, and open risks
- include capability snapshot, session summary, status card, shape-advisor output, operator brief, run handles, source freshness status

## Required Deliverables

- route decision (`workflow_id`, `execution_locus`, `session_handoff_state`)
- capability snapshot
- session status summary (`running_jobs`, `queued_jobs`, `eqw_jobs`, threshold state)
- active-job snapshot (`job_id`, `state`, `queue`, `slots`, `task_id`)
- status card (`Health`, `Execution Locus`, `Running Jobs`, `Queued Jobs`, `Eqw Jobs`, `Reason`, `Recommendation`)
- submission-shape advisor report (`advisor`, `reason`, `queue_policy`, `recommended_action`)
- operator brief report (`submit_gate`, `advisor`, `next_action`, `queue_policy`)
- submit gate checklist and qa preflight report per artifact
- run handles for each submitted job (job id, state, log path, timestamps)
- source table entries for external policy claims plus freshness state
- assumption and clarification log

## Output Contract

Return:
1. Decision summary
- scheduler context, selected route, execution locus, risk posture

2. Capability evidence
- probe/session outputs summarized into capability snapshot
- include status card from `references/user-status-contract.md`

3. Execution plan
- exact commands in preflight, submit, verify order
- include qa preflight, session status, shape-advisor, and operator-brief outcomes
- if `running_jobs > 3`, include warning and explicit confirmation checkpoint before additional submit commands
- include queue fairness note: no queue bypass and no line-skipping behavior

4. Runtime status
- run handles and next actions for each submitted job

5. Follow-up
- unresolved unknowns, stale-policy remediations, and next checks

## Trigger Tests

Should trigger:
- "start a densegen workspace x batch job on bu scc"
- "also track and wire up notify for slack notifications"
- "submit a request for an interactive on demand session"
- "i've just entered into an ondemand session, do the following task"
- "Submit this qsub job on BU SCC."
- "Help me debug an Eqw job and recover safely."
- "This job keeps getting killed by the process reaper."
- "Run a DenseGen workflow for stress ethanol and cipro workspace for two hours."
- "Start DenseGen batch for stress ethanol and cipro without notify."
- "Show my active SGE jobs and tell me if I should submit more right now."

Should not trigger:
- "Write a Flask endpoint."
- "Fix this pandas DataFrame bug."
- "Tune a Kubernetes deployment."
- "Do a local shell refactor with no scheduler involvement."

## Examples

- "Show my active BU SCC jobs, tell me whether I should submit more now, and recommend arrays vs `-hold_jid`"; "Plan a DenseGen batch submit with Notify wiring, but stop at the verify-before-submit gate"; "I am already inside OnDemand; probe the session, report status, and continue with the next SGE step safely."

## Troubleshooting

- Undertriggering: expand scheduler, notify, and OnDemand handoff phrases in description.
- Overtriggering: tighten non-scheduler coding boundaries in description and trigger tests.
- Missing capability facts: rerun probes and regenerate snapshot before proposing commands.
- Queue churn or duplicate submissions: enforce submit fingerprint, queue-pressure warning, shape advisor output, and operator brief gate.
- Handoff confusion: if user is already in OnDemand, skip session-request flow and continue in-session.
- Stale policy assumptions: refresh from official BU pages and update source evidence table.

## Additional Resources

- `references/README.md`
- `references/route-load-matrix.md`, `references/probe-first-contract.md`, and
  `references/workflow-router.md`
- `references/session-status-reporting.md`, `references/user-status-contract.md`,
  `references/submission-shape-advisor.md`, and `references/operator-brief.md`
- `references/interactive-contract.md`, `references/batch-submit-contract.md`,
  `references/runbook-entrypoints.md`, and
  `references/automation-qa-preflight.md`
- `references/ci-mechanical-gates.md`, `references/workload-dnadesign.md`,
  `references/bu-scc-system-usage.md`, `references/source-evidence.md`, and
  `references/test-matrix.md`
- `scripts/qa-sge-submit-preflight.sh`, `scripts/sge-session-status.sh`,
  `scripts/sge-active-jobs.sh`, `scripts/sge-status-card.sh`,
  `scripts/sge-submit-shape-advisor.sh`, and `scripts/sge-operator-brief.sh`
- Run `scripts/audit-sge-hpc-ops-skill.sh` for deterministic skill-contract checks.
