## Ops operations index

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-03

Ops is the repository-level orchestration surface for deterministic batch workflows. This page is a route map; detailed command behavior and schema rules live in [orchestration runbooks](orchestration-runbooks.md).

### What Ops is for

1. Turn runbook intent into deterministic preflight, verification, and submit phases.
2. Keep runbook, scheduler-log, and audit artifacts workspace-scoped for repeated campaigns.
3. Fail fast on schema, secret, and storage-guard violations before submission.
4. Produce machine-readable audit output that records command order and outcomes.

### Start here

1. Start with [Ops package README](../../src/dnadesign/ops/README.md) for scope and boundaries.
2. Choose a route in [Workflow routes](#workflow-routes) based on batch intent.
3. Confirm contract details in [Contracts](#contracts).
4. Run the [Verification loop](#verification-loop) before any submit.
5. Return to the [repository docs index](../README.md) for cross-tool routing.

### Workflow routes

| Need | Start here | Verify next |
| --- | --- | --- |
| Bootstrap a runbook from scratch | [runbook bootstrap path](orchestration-runbooks.md#runbook-bootstrap-path) | [runbook schema (v1)](orchestration-runbooks.md#runbook-schema-v1) |
| Validate command order without side effects | [2-minute dry-run path](orchestration-runbooks.md#2-minute-dry-run-path) | [contract rules](orchestration-runbooks.md#contract-rules) |
| Run batch-only orchestration | [workflow routes](orchestration-runbooks.md#workflow-routes) | [planner and executor commands](orchestration-runbooks.md#planner-and-executor-commands) |
| Run batch plus notify orchestration | [workflow routes](orchestration-runbooks.md#workflow-routes) | [notify command contracts](../../src/dnadesign/notify/docs/reference/command-contracts.md) |
| Run generation now and refresh plots in the same submit chain | [runbook schema (v1)](orchestration-runbooks.md#runbook-schema-v1) | [contract rules](orchestration-runbooks.md#contract-rules) |

### Contracts

1. [runbook init command contract](orchestration-runbooks.md#runbook-bootstrap-path)
2. [runbook plan command contract](orchestration-runbooks.md#planner-and-executor-commands)
3. [runbook execute command contract](orchestration-runbooks.md#planner-and-executor-commands)
4. [Runbook schema (v1)](orchestration-runbooks.md#runbook-schema-v1)
5. [Contract rules](orchestration-runbooks.md#contract-rules)
6. [Packaged runbook precedents](../../src/dnadesign/ops/runbooks/presets)

### Verification loop

1. Create or validate runbook shape with `uv run ops runbook init --workflow <workflow> ...`.
2. Render deterministic commands with `uv run ops runbook plan --runbook <runbook.yaml> --repo-root <repo-root>`.
3. Execute dry gates with `uv run ops runbook execute --runbook <runbook.yaml> --repo-root <repo-root> --audit-json <audit.json> --no-submit`.
4. Review audit JSON fields (`execution.ok`, `execution.failed_phase`, ordered command records).
5. Submit only after dry gates remain green.

### Operator quickstart

```bash
uv run ops runbook init --workflow <workflow> --runbook <runbook.yaml> --workspace-root <workspace-root> --repo-root <repo-root> --project dunlop --id <runbook-id>
uv run ops runbook plan --runbook <runbook.yaml> --repo-root <repo-root>
uv run ops runbook execute --runbook <runbook.yaml> --repo-root <repo-root> --audit-json <workspace-root>/outputs/logs/ops/audit/<file>.json --no-submit
uv run ops runbook execute --runbook <runbook.yaml> --repo-root <repo-root> --audit-json <workspace-root>/outputs/logs/ops/audit/<file>.json --submit
```

- Keep runbooks workspace-scoped (for example `<workspace-root>/outputs/logs/ops/runbooks/<runbook-id>.yaml`).
- Do not create transient operational working directories at repo root (`.codex_tmp/`, `.tmp_ops/`, `tmp_ops/`); use `/scratch` for disposable state.
- For manual chaining, `--active-job-id` accepts repeat flags or a comma-delimited list and normalizes before `-hold_jid` submit wiring.
- `ops runbook active-jobs` returns `plan_command_hint` and active-job arg hints so you can paste manual chaining arguments directly.
- Notify-enabled routes require a readable webhook file contract before `ops runbook execute`:
  `NOTIFY_WEBHOOK_FILE` (`<webhook_env>_FILE`) or a profile webhook `secret_ref` that resolves to `file://...`.
