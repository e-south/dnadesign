## Route Load Matrix

Progressive-disclosure router for `sge-hpc-ops`.

Load policy:
- Always load base pack first.
- Load exactly one route pack from this matrix.
- Add optional packs only when the workflow needs them.

### Base pack (always)

- `probe-first-contract.md`
- `workflow-router.md`

### Cross-cutting packs (conditional)

Load when needed by task scope:
- user status communication: `session-status-reporting.md`, `user-status-contract.md`
- submit-shape and submit readiness: `submission-shape-advisor.md`, `operator-brief.md`
- command-first Ops runbooks: `runbook-entrypoints.md`
- BU SCC policy claims: `bu-scc-system-usage.md`, `source-evidence.md`
- automation/policy mechanics: `ci-mechanical-gates.md`

### Route packs

`densegen_batch_submit`
- `batch-submit-contract.md`
- `automation-qa-preflight.md`
- `workload-dnadesign.md`

`densegen_batch_with_notify_slack`
- `batch-submit-contract.md`
- `automation-qa-preflight.md`
- `workload-dnadesign.md`

`ondemand_session_request`
- `interactive-contract.md`
- `automation-qa-preflight.md`

`ondemand_session_handoff`
- `interactive-contract.md`

`generic_sge_ops`
- `batch-submit-contract.md` or `interactive-contract.md` (choose one based on intent)

### Script map by concern

status and pressure:
- `scripts/sge-session-status.sh`
- `scripts/sge-active-jobs.sh`
- `scripts/sge-status-card.sh`

submit-shape and readiness:
- `scripts/sge-submit-shape-advisor.sh`
- `scripts/sge-operator-brief.sh`

submit artifact checks:
- `scripts/qa-sge-submit-preflight.sh`

### Minimal loading examples

DenseGen batch with Notify:
1. base pack
2. route pack: `densegen_batch_with_notify_slack`
3. cross-cutting: status + submit-shape + BU SCC policy (if BU context)

OnDemand handoff (already in session):
1. base pack
2. route pack: `ondemand_session_handoff`
3. cross-cutting: status pack only if user asks for queue pressure or submit readiness
