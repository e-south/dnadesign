## Ops orchestration index

**Type:** route
**Plane:** control-plane
**Owner-boundary:** ops
**Entry artifact:** batch orchestration intent that still needs a runbook lifecycle route
**Exit artifact:** authoritative ops schema, plan, or execute contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-16

Ops is the repository-level control-plane orchestration surface for deterministic batch workflows. This page is a route map; detailed command behavior and schema rules live in [orchestration runbooks](orchestration-runbooks.md). It is not the registry for durable USR-backed data-plane workflows.

### What Ops is for

1. Turn runbook intent into deterministic preflight, verification, and submit phases.
2. Keep runbook, scheduler-log, and audit artifacts workspace-scoped for repeated campaigns.
3. Fail fast on schema, secret, and storage-guard violations before submission.
4. Produce machine-readable audit output that records command order and outcomes.

### Start here

1. Start with [Ops package README](../../src/dnadesign/ops/README.md) for scope and boundaries.
2. Choose a route in [Orchestration routes](#orchestration-routes) based on batch intent.
3. Confirm contract details in [Contracts](#contracts).
4. Run the [Verification loop](#verification-loop) before any submit.
5. Return to the [repository docs index](../README.md) for cross-tool routing across control-plane and data-plane surfaces.

### Orchestration routes

| Need | Start here | Verify next |
| --- | --- | --- |
| Bootstrap a runbook from scratch | [runbook bootstrap path](orchestration-runbooks.md#runbook-bootstrap-path) | [runbook schema (v1)](orchestration-runbooks.md#runbook-schema-v1) |
| Validate command order without side effects | [2-minute dry-run path](orchestration-runbooks.md#2-minute-dry-run-path) | [contract rules](orchestration-runbooks.md#contract-rules) |
| Run batch-only orchestration | [orchestration workflow ids](orchestration-runbooks.md#orchestration-workflow-ids) | [planner and executor commands](orchestration-runbooks.md#planner-and-executor-commands) |
| Run batch plus notify orchestration | [orchestration workflow ids](orchestration-runbooks.md#orchestration-workflow-ids) | [notify command contracts](../../src/dnadesign/notify/docs/reference/command-contracts.md) |
| Run generation now and refresh plots in the same submit chain | [runbook schema (v1)](orchestration-runbooks.md#runbook-schema-v1) | [contract rules](orchestration-runbooks.md#contract-rules) |

### Adjacent routes outside Ops

Ops does not own construct-led source-of-truth accumulation or other USR-backed data-plane procedures. When you want a shared USR-backed dataset that multiple construct projects feed before infer adds derived namespaces, use the shared cross-tool runbook:

- [Multi-source source-of-truth assembly](../../src/dnadesign/usr/docs/operations/multi-source-source-of-truth-assembly.md)
- [Construct -> USR -> Infer source-of-truth runbook](../../src/dnadesign/usr/docs/operations/construct-infer-source-of-truth-runbook.md)
- [Promoter characterization feature matrix](../../src/dnadesign/usr/docs/operations/promoter-characterization-feature-matrix.md)

### Contracts

1. [runbook init command contract](orchestration-runbooks.md#runbook-bootstrap-path)
2. [runbook plan command contract](orchestration-runbooks.md#planner-and-executor-commands)
3. [runbook execute command contract](orchestration-runbooks.md#planner-and-executor-commands)
4. [Runbook schema (v1)](orchestration-runbooks.md#runbook-schema-v1)
5. [Contract rules](orchestration-runbooks.md#contract-rules)
6. [Packaged runbook presets](../../src/dnadesign/ops/runbooks/presets)

### Verification loop

1. Create or validate runbook shape with `uv run ops runbook init --workflow <workflow> ...`.
2. Render deterministic commands with `uv run ops runbook plan --runbook <runbook.yaml> --repo-root <repo-root>`.
3. Execute dry gates with `uv run ops runbook execute --runbook <runbook.yaml> --repo-root <repo-root> --audit-json <audit.json> --no-submit`.
4. Review audit JSON fields (`execution.ok`, `execution.failed_phase`, ordered command records).
5. Submit only after dry gates remain green.

### Operator quickstart

```bash
uv run ops runbook init --workflow <workflow> --runbook <runbook.yaml> --workspace-root <workspace-root> --repo-root <repo-root> --project <project> --id <runbook-id>
uv run ops runbook plan --runbook <runbook.yaml> --repo-root <repo-root>
uv run ops runbook execute --runbook <runbook.yaml> --repo-root <repo-root> --audit-json <workspace-root>/outputs/logs/ops/audit/<file>.json --no-submit
uv run ops runbook execute --runbook <runbook.yaml> --repo-root <repo-root> --audit-json <workspace-root>/outputs/logs/ops/audit/<file>.json --submit
```

- Keep runbooks workspace-scoped (for example `<workspace-root>/outputs/logs/ops/runbooks/<runbook-id>.yaml`).
- Keep `<project>` aligned with the scheduler account or project configured for the workspace or study.
- Do not create transient operational working directories at repo root (`.codex_tmp/`, `.tmp_ops/`, `tmp_ops/`); use `/scratch` for disposable state.
- For manual chaining, `--active-job-id` accepts repeat flags or a comma-delimited list and normalizes before `-hold_jid` submit wiring.
- `ops runbook active-jobs` returns `plan_command_hint` and active-job arg hints so you can paste manual chaining arguments directly.
- Notify-enabled routes require a readable webhook file contract before `ops runbook execute`:
  `NOTIFY_WEBHOOK_FILE` (`<webhook_env>_FILE`) or a profile webhook `secret_ref` that resolves to `file://...`.
