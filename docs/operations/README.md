## Ops orchestration index

**Type:** route
**Plane:** control-plane
**Owner-boundary:** ops
**Entry artifact:** batch job setup that still needs an Ops runbook route
**Exit artifact:** Ops schema, plan, execute, or read-only progress docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-20

Ops covers repo-level batch orchestration for deterministic workflows. Start here when you need runbook setup, planning, execution, or status checks. Detailed command behavior and schema rules live in [orchestration runbooks](orchestration-runbooks.md). For command lookup across tools, use the [runbook catalog](../runbooks/README.md).

### What Ops is for

1. Turn runbook intent into deterministic preflight, verification, and submit phases.
2. Keep runbook, scheduler-log, and audit artifacts workspace-scoped for repeated campaigns.
3. Fail fast on schema, secret, and storage-guard violations before submission.
4. Produce machine-readable audit output that records command order and outcomes.
5. Expose read-only progress summaries over registered procedures and explicit campaigns without taking ownership away from boundary-local tools.

### Start here

1. Start with [Ops package README](../../src/dnadesign/ops/README.md) for scope and boundaries.
2. Use [Discovery handoff](#discovery-handoff) when you still need the catalog or status commands.
3. Choose a route in [Orchestration routes](#orchestration-routes) based on batch intent.
4. Confirm contract details in [Contracts](#contracts).
5. Use [Status and manifest routes](#status-and-manifest-routes) when you need status or an explicit manifest rather than command planning.
6. Run the [Verification loop](#verification-loop) before any submit.
7. Return to the [repository docs index](../README.md) if the next step is outside Ops.

### Discovery handoff

Use these when you still need command lookup before choosing a runbook lifecycle step.

- [Runbook catalog](../runbooks/README.md): shared command table for `ops catalog` and `ops progress`.
- [How to use Ops](../../src/dnadesign/ops/docs/how-to-use-ops.md): package command guide for inspection, status, and manifest commands.
- `uv run ops catalog list --simple`: quick inventory command before choosing a lifecycle route.

### Orchestration routes

| Need | Start here | Verify next |
| --- | --- | --- |
| Bootstrap a runbook from scratch | [runbook bootstrap path](orchestration-runbooks.md#runbook-bootstrap-path) | [runbook schema (v1)](orchestration-runbooks.md#runbook-schema-v1) |
| Validate command order without side effects | [2-minute dry-run path](orchestration-runbooks.md#2-minute-dry-run-path) | [contract rules](orchestration-runbooks.md#contract-rules) |
| Run batch-only orchestration | [orchestration workflow ids](orchestration-runbooks.md#orchestration-workflow-ids) | [planner and executor commands](orchestration-runbooks.md#planner-and-executor-commands) |
| Run batch plus notify orchestration | [orchestration workflow ids](orchestration-runbooks.md#orchestration-workflow-ids) | [notify command contracts](../../src/dnadesign/notify/docs/reference/command-contracts.md) |
| Run generation now and refresh plots in the same submit chain | [runbook schema (v1)](orchestration-runbooks.md#runbook-schema-v1) | [contract rules](orchestration-runbooks.md#contract-rules) |

### Adjacent routes outside Ops

Ops does not own construct-led source-of-truth accumulation or other USR-backed workflows. When you want a shared USR-backed dataset that multiple construct projects feed before infer adds derived namespaces, use the shared runbook:

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

### Status and manifest routes

1. Use `uv run ops progress explain <registry-id>` to see the required flags and a ready-to-paste `progress show` command before you touch artifacts.
2. Use `uv run ops progress show ops.control-plane.orchestration --repo-root <repo-root> --audit-json <workspace-root>/outputs/logs/ops/audit/<file>.json` to summarize one control-plane runbook execution from the registered progress contract.
3. Use `uv run ops progress show usr.data-plane.promoter-feature-matrix --repo-root <repo-root> --usr-root <usr-root> --dataset <dataset>` to summarize one staged USR-backed data-plane procedure from explicit artifacts.
4. `ops progress show` and `ops progress campaign` are read-only status surfaces. Inspect the required flags in `ops progress explain <registry-id>` or `ops catalog show <registry-id>` before you run them if you do not already know the artifact contract.
5. Use `uv run ops progress scaffold ops.control-plane.orchestration usr.data-plane.promoter-feature-matrix --repo-root <repo-root>` to emit a manifest template with the right required fields. It prints to stdout unless you pass `--out`.
6. Use `uv run ops progress scaffold --related-to usr.data-plane.promoter-feature-matrix --repo-root <repo-root>` when you want the named registered procedure plus its related procedures as a starting point.
7. Use `uv run ops progress campaign --repo-root <repo-root> --manifest <manifest.yaml>` when the work spans multiple runtimes or pauses between steps.
8. Keep the manifest explicit. Ops reads the files you name there; it does not infer hidden campaign state.
9. For progress-kind meanings and the next checks for each one, return to the [runbook catalog glossary](../runbooks/README.md#progress-surface-glossary).

### Verification loop

1. Create or validate runbook shape with `uv run ops runbook init --workflow <workflow> ...`.
2. Render deterministic commands with `uv run ops runbook plan --runbook <runbook.yaml> --repo-root <repo-root>`.
3. Execute dry gates with `uv run ops runbook execute --runbook <runbook.yaml> --repo-root <repo-root> --audit-json <audit.json> --no-submit`.
   On workstations without `qstat`, add `--allow-missing-qstat`; the queue probe remains explicit and the resulting audit will summarize as attention rather than hiding the degraded state.
4. Review audit JSON fields (`execution.ok`, `execution.failed_phase`, ordered command records).
5. Optionally summarize the latest runbook state with `uv run ops progress show ops.control-plane.orchestration --repo-root <repo-root> --audit-json <workspace-root>/outputs/logs/ops/audit/<file>.json`.
6. Submit only after dry gates remain green.

### Operator quickstart

```bash
uv run ops runbook init --workflow <workflow> --runbook <runbook.yaml> --workspace-root <workspace-root> --repo-root <repo-root> --project <project> --id <runbook-id>
uv run ops runbook plan --runbook <runbook.yaml> --repo-root <repo-root>
uv run ops runbook execute --runbook <runbook.yaml> --repo-root <repo-root> --audit-json <workspace-root>/outputs/logs/ops/audit/<file>.json --no-submit
uv run ops runbook execute --runbook <runbook.yaml> --repo-root <repo-root> --audit-json <workspace-root>/outputs/logs/ops/audit/<file>.json --submit
uv run ops progress explain ops.control-plane.orchestration
uv run ops progress show ops.control-plane.orchestration --repo-root <repo-root> --audit-json <workspace-root>/outputs/logs/ops/audit/<file>.json
uv run ops progress scaffold ops.control-plane.orchestration usr.data-plane.promoter-feature-matrix --repo-root <repo-root>
uv run ops progress scaffold --related-to usr.data-plane.promoter-feature-matrix --repo-root <repo-root>
uv run ops progress campaign --repo-root <repo-root> --manifest <manifest.yaml>
```

- Keep runbooks workspace-scoped (for example `<workspace-root>/outputs/logs/ops/runbooks/<runbook-id>.yaml`).
- The dry run above is the smallest working status example because it emits the audit JSON that `ops progress show ops.control-plane.orchestration` reads. On non-SCC workstations, add `--allow-missing-qstat` so queue readiness degrades explicitly instead of failing opaquely.
- Keep `<project>` aligned with the scheduler account or project configured for the workspace or study.
- Do not create transient operational working directories at repo root (`.codex_tmp/`, `.tmp_ops/`, `tmp_ops/`); use `/scratch` for disposable state.
- For manual chaining, `--active-job-id` accepts repeat flags or a comma-delimited list and normalizes before `-hold_jid` submit wiring.
- `ops runbook active-jobs` returns `plan_command_hint` and active-job arg hints so you can paste manual chaining arguments directly.
- Notify-enabled routes require a readable webhook file contract before `ops runbook execute`:
  `NOTIFY_WEBHOOK_FILE` (`<webhook_env>_FILE`) or a profile webhook `secret_ref` that resolves to `file://...`.
