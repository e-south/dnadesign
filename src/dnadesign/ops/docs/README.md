## Ops docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-27

Ops package docs cover packaged presets and command reference. The repo-wide
command list lives in `docs/runbooks/README.md`.

### Start here

- [Runbook catalog](../../../../docs/runbooks/README.md): repo-level command index when you want the procedure list first.
- [How to use Ops](how-to-use-ops.md): command guide for catalog discovery, procedure inspection, status checks, and manifest scaffolds.
- [Ops orchestration index](../../../../docs/operations/README.md): docs for init, plan, execute, and verification.
- [OPS mental model](../../../../docs/operations/ops-mental-model.md): shortest correct model for planes, state semantics, and snapshot versus preflight.
- [OPS failure contract](../../../../docs/operations/ops-failure-contract.md): exit-code and stderr contract for CLI and automation consumers.
- [OPS status kinds](../../../../docs/operations/ops-status-kinds.md): registry ids, status kinds, owners, scope, and required inputs.
- [OPS preflight checks](../../../../docs/operations/ops-preflight-checks.md): generic readiness-check vocabulary used by `ops.study.yaml`.
- [Orchestration runbooks](../../../../docs/operations/orchestration-runbooks.md): runbook schema, command order, and execution contracts.
- [Repository docs index](../../../../docs/README.md): docs index when the next step is outside Ops.

If you are entering from the shell rather than browsing docs first, start with `uv run ops catalog list --simple`, then open [How to use Ops](how-to-use-ops.md) for the command summary.

### Operator overlays

- [SGE/HPC operator skill](../../../../.agents/skills/sge-hpc-ops/SKILL.md): use when the task is scheduler-shaped (`qsub`, `qrsh`, queue state, BU SCC execution posture, or batch/Notify chains).
- [Notify operator skill](../../../../.agents/skills/notify-ops/SKILL.md): use when the task is Slack watcher setup, validation, live watch, or recovery.
- [BU SCC USR sync skill](../../../../.agents/skills/bu-scc-usr-sync/SKILL.md): use when the task is dataset sync between local dnadesign and BU SCC.

### Package-local surfaces

- [Packaged runbook presets](../runbooks/presets): checked-in starter runbooks for common orchestration routes.
- [`dnadesign.ops`](../README.md): package README with the short tool overview.
- [Runbook catalog status views](../../../../docs/runbooks/README.md#status-views): glossary for registered status adapters and explicit campaign manifests.

### Boundary reminders

- Ops owns control-plane orchestration, audit trails, and scheduler command ordering.
- `ops progress show` and `ops progress campaign` are read-only; `ops progress scaffold` prints YAML to stdout unless you pass `--out`.
- Tool-specific runtime semantics stay in the boundary-owning tool docs.
- Durable USR-backed data-plane workflows stay in shared USR operations docs.
