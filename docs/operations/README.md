## Ops orchestration index

**Type:** route
**Plane:** control-plane
**Owner-boundary:** ops
**Entry artifact:** batch job setup that still needs an Ops runbook route
**Exit artifact:** Ops schema, plan, execute, or read-only progress docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-21

Use this page when the next step is batch orchestration or a read-only Ops status check. If the work is dataset assembly, construct realization, or infer write-back, leave this page and continue in the shared USR runbooks. Detailed command behavior and schema rules live in [orchestration runbooks](orchestration-runbooks.md). Use the [runbook catalog](../runbooks/README.md) when you need command lookup across tools.

### What Ops is for

- Turn runbook intent into deterministic preflight, verification, and submit phases.
- Keep runbook, scheduler-log, and audit artifacts workspace-scoped for repeated campaigns.
- Fail fast on schema, secret, and storage-guard violations before submission.
- Produce machine-readable audit output that records command order and outcomes.
- Expose read-only status summaries over registered procedures and explicit campaigns without taking ownership away from boundary-local tools.

### Start here

1. Use [Command lookup](#command-lookup) when you need the catalog or status commands first.
2. Use [Orchestration routes](#orchestration-routes) when you are starting, dry-running, or submitting a runbook.
3. Use [Contracts](#contracts) when you need the exact schema or command rules.
4. Use [Status and manifest routes](#status-and-manifest-routes) when you need a read-only summary or an explicit manifest.
5. Run the [Verification loop](#verification-loop) before any submit.
6. Leave Ops for the shared USR runbooks when the next step changes datasets rather than scheduler state.

### Command lookup

Use these when you still need command lookup before choosing a runbook lifecycle step.

- [Runbook catalog](../runbooks/README.md): shared command index for `ops catalog` and `ops progress`.
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
4. Use `uv run ops progress show usr.data-plane.promoter-study-status` when you want the one-command summary of the active checked-in live study before drilling into tool-local status. Add `--repo-root <repo-root> --study-dir docs/studies/<study-id>` when you need to pin a different study or invoke it from outside the repo checkout.
5. Use `uv run ops progress show usr.data-plane.promoter-study-preflight` when you need the deeper read-only command preflight across DenseGen, Construct, Infer, Notify, and batch-plan surfaces for that same study.
6. `ops progress show` and `ops progress campaign` are read-only status commands. Inspect the required flags in `ops progress explain <registry-id>` or `ops catalog show <registry-id>` before you run them if you do not already know the artifact contract.
7. Use `uv run ops progress scaffold ops.control-plane.orchestration usr.data-plane.promoter-feature-matrix --repo-root <repo-root>` to emit a manifest template with the right required fields. It prints to stdout unless you pass `--out`.
8. Use `uv run ops progress scaffold --related-to usr.data-plane.promoter-feature-matrix --repo-root <repo-root>` when you want the named registered procedure plus its related procedures as a starting point.
9. Use `uv run ops progress campaign --repo-root <repo-root> --manifest <manifest.yaml>` when the work spans multiple runtimes or pauses between steps.
10. Keep the manifest explicit. Ops reads the files you name there; it does not infer hidden campaign state.
11. For live promoter-study status, keep the study files under `docs/studies/<study-id>/`. Use [Study records index](../studies/README.md) for the required layout and selector rules.
12. For status-kind meanings and the next checks for each one, see the [runbook catalog status views](../runbooks/README.md#status-views).
13. If the next step is dataset assembly, construct realization, or infer write-back, leave Ops and continue in the shared USR runbooks:
    [Multi-source shared dataset assembly](../../src/dnadesign/usr/docs/operations/multi-source-shared-dataset-assembly.md),
    [Construct -> USR -> Infer shared dataset runbook](../../src/dnadesign/usr/docs/operations/construct-infer-shared-dataset-runbook.md),
    or [Promoter characterization feature matrix](../../src/dnadesign/usr/docs/operations/promoter-characterization-feature-matrix.md).

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
