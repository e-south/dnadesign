## Ops orchestration index

**Type:** route
**Plane:** control-plane
**Owner-boundary:** ops
**Entry artifact:** batch job setup that still needs an Ops runbook route
**Exit artifact:** Ops schema, plan, execute, or read-only progress docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-26

Batch orchestration and read-only Ops status checks start here. Dataset
assembly, construct realization, and infer write-back stay in the shared USR
runbooks. Detailed command behavior and schema rules live in
[orchestration runbooks](orchestration-runbooks.md); command lookup lives in
the [runbook catalog](../runbooks/README.md).

### OPS docs

- [OPS mental model](ops-mental-model.md): one-page plane model, state semantics, snapshot vs preflight, and source-of-truth map.
- [OPS failure contract](ops-failure-contract.md): exit codes, stderr rules, and maintainer-facing failure expectations.
- [OPS runtime visibility](ops-runtime-visibility.md): scheduler probe states, active-job resolution states, and degraded submit rules.
- [OPS status kinds](ops-status-kinds.md): public routes, status kinds, owners, scope, and required inputs.
- [OPS preflight checks](ops-preflight-checks.md): generic readiness check vocabulary used by `ops.study.yaml`.
- [Orchestration runbooks](orchestration-runbooks.md): runbook schema, planner, executor, and scheduler-facing contracts.

### What Ops is for

- Turn runbook intent into deterministic preflight, verification, and submit phases.
- Keep runbook, scheduler-log, and audit artifacts workspace-scoped for repeated campaigns.
- Fail fast on schema, secret, and storage-guard violations before submission.
- Produce machine-readable audit output that records command order and outcomes.
- Expose read-only status summaries over registered procedures and explicit campaigns without taking ownership away from boundary-local tools.

### Start here

1. Use [Command lookup](#command-lookup) when you need the catalog or status commands first.
2. Read the [OPS mental model](ops-mental-model.md) if you need the plane model or state lattice first.
3. Use [Orchestration routes](#orchestration-routes) when you are starting, dry-running, or submitting a runbook.
4. Use [Contracts](#contracts) when you need the exact schema or command rules.
5. Use [Status and manifest routes](#status-and-manifest-routes) when you need a read-only summary or an explicit manifest.
6. Run the [Verification loop](#verification-loop) before any submit.
7. Leave Ops for the shared USR runbooks when the next step changes datasets rather than scheduler state.

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
4. Use `uv run ops progress show studies.stress-ethanol-cipro-growth.status` only for the concrete `stress_ethanol_cipro_growth` study before drilling into tool-local status.
5. Use `uv run ops progress show studies.stress-ethanol-cipro-growth.preflight --scope next --command-timeout-seconds 30 --json` when you need the deeper execution-readiness blockers for that study. The surface reports check kinds declared in its `ops.study.yaml`, not hidden family-only readiness branches.
6. `ops progress show` and `ops progress campaign` are read-only status commands. Inspect the required flags in `ops progress explain <registry-id>` or `ops catalog show <registry-id>` before you run them if you do not already know the artifact contract.
7. Use `uv run ops progress scaffold ops.control-plane.orchestration usr.data-plane.promoter-feature-matrix --repo-root <repo-root>` to emit a manifest template with the right required fields. It prints to stdout unless you pass `--out`.
8. Use `uv run ops progress scaffold --related-to usr.data-plane.promoter-feature-matrix --repo-root <repo-root>` when you want the named registered procedure plus its related procedures as a starting point.
9. Use `uv run ops progress campaign --repo-root <repo-root> --manifest <manifest.yaml>` when the work spans multiple runtimes or pauses between steps.
10. Keep the manifest explicit. Ops reads the files you name there; it does not infer hidden campaign state.
11. For study status, use the concrete study-owned surface only when one exists. For `stress_ethanol_cipro_growth`, keep the study files under `docs/studies/stress_ethanol_cipro_growth/` and use [Study records index](../studies/README.md) for selector rules.
12. For status-kind meanings and the next checks for each one, see the [runbook catalog status views](../runbooks/README.md#status-views).
13. If the next step is dataset assembly, construct realization, or infer write-back, leave Ops and continue in the shared USR runbooks:
    [Multi-source shared dataset assembly](../../src/dnadesign/usr/docs/operations/multi-source-shared-dataset-assembly.md),
    [Construct -> USR -> Infer shared dataset runbook](../../src/dnadesign/usr/docs/operations/construct-infer-shared-dataset-runbook.md),
    or [Promoter characterization feature matrix](../../src/dnadesign/usr/docs/operations/promoter-characterization-feature-matrix.md).

### Verification loop

1. Create or validate runbook shape with `uv run ops runbook init --workflow <workflow> ... --project <project>` or an explicit preset such as `--preset bu-scc-dunlop`.
2. Render deterministic commands with `uv run ops runbook plan --runbook <runbook.yaml> --repo-root <repo-root>`.
3. Execute dry gates with `uv run ops runbook execute --runbook <runbook.yaml> --repo-root <repo-root> --audit-json <audit.json> --no-submit`.
   On workstations without `qstat`, add `--allow-missing-qstat`; the queue probe remains explicit and the resulting audit will summarize as attention rather than hiding the degraded state.
4. Review audit JSON fields (`plan.runtime_visibility`, `plan.warnings`, `execution.ok`, `execution.failed_phase`, ordered command records).
5. Optionally summarize the latest runbook state with `uv run ops progress show ops.control-plane.orchestration --repo-root <repo-root> --audit-json <workspace-root>/outputs/logs/ops/audit/<file>.json`.
6. Submit only after dry gates remain green. If `runtime_visibility.active_job_resolution_state=unknown`, `ops runbook execute --submit` fails closed unless you pass `--allow-unknown-active-jobs`.

### Operator quickstart

```bash
uv run ops runbook init --workflow <workflow> --runbook <runbook.yaml> --workspace-root <workspace-root> --repo-root <repo-root> --project <project> --id <runbook-id>
uv run ops runbook init --workflow <workflow> --runbook <runbook.yaml> --workspace-root <workspace-root> --repo-root <repo-root> --preset bu-scc-dunlop --id <runbook-id>
uv run ops runbook plan --runbook <runbook.yaml> --repo-root <repo-root>
uv run ops runbook execute --runbook <runbook.yaml> --repo-root <repo-root> --audit-json <workspace-root>/outputs/logs/ops/audit/<file>.json --no-submit
uv run ops runbook execute --runbook <runbook.yaml> --repo-root <repo-root> --audit-json <workspace-root>/outputs/logs/ops/audit/<file>.json --submit
uv run ops runbook diagnostics session-counts --qstat-file <fixture>
uv run ops runbook diagnostics submit-shape-advisor --qstat-file <fixture> --planned-submits <N> --warn-over-running 3
uv run ops runbook diagnostics operator-brief --qstat-file <fixture> --planned-submits <N> --warn-over-running 3
uv run ops progress explain ops.control-plane.orchestration
uv run ops progress show ops.control-plane.orchestration --repo-root <repo-root> --audit-json <workspace-root>/outputs/logs/ops/audit/<file>.json
uv run ops progress scaffold ops.control-plane.orchestration usr.data-plane.promoter-feature-matrix --repo-root <repo-root>
uv run ops progress scaffold --related-to usr.data-plane.promoter-feature-matrix --repo-root <repo-root>
uv run ops progress campaign --repo-root <repo-root> --manifest <manifest.yaml>
```

- Keep runbooks workspace-scoped (for example `<workspace-root>/outputs/logs/ops/runbooks/<runbook-id>.yaml`).
- The dry run above is the smallest working status example because it emits the audit JSON that `ops progress show ops.control-plane.orchestration` reads. On non-SCC workstations, add `--allow-missing-qstat` so queue readiness degrades explicitly instead of failing opaquely.
- Keep `<project>` aligned with the scheduler account or project configured for the workspace or study. Presets are explicit shortcuts, not hidden defaults.
- Do not create transient operational working directories at repo root (`.codex_tmp/`, `.tmp_ops/`, `tmp_ops/`); use `/scratch` for disposable state.
- For manual chaining, `--active-job-id` accepts repeat flags or a comma-delimited list and normalizes before `-hold_jid` submit wiring.
- `ops runbook active-jobs` returns `runtime_visibility`, `plan_command_hint`, and active-job arg hints so you can paste manual chaining arguments directly.
- `ops runbook plan` may still return a usable dry-run plan when runtime visibility is degraded, but `ops runbook execute --submit` blocks by default when active-job posture is unknown.
- Notify-enabled routes require a readable webhook file contract before `ops runbook execute`:
  `NOTIFY_WEBHOOK_FILE` (`<webhook_env>_FILE`) or a profile webhook `secret_ref` that resolves to `file://...`.
