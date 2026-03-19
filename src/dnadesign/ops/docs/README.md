## ops docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-19

Use this index for package-local Ops documentation. Shared operator procedures stay under top-level `docs/operations/` because Ops owns the control-plane runbook surface for the repository.

### Start here

- [Runbook catalog](../../../../docs/runbooks/README.md): repo-level inventory when you want a concise list of authoritative procedures first.
- `uv run ops catalog list`: terminal inventory for the same shared runbook catalog when you are already in the shell, with filters such as `--plane data-plane --query infer`.
- [Ops orchestration index](../../../../docs/operations/README.md): task-first router for init, plan, execute, and verification flows.
- [Orchestration runbooks](../../../../docs/operations/orchestration-runbooks.md): authoritative runbook schema, command order, and execution contracts.
- [Repository docs index](../../../../docs/README.md): repo-wide route map when the next step is outside Ops.

### Package-local surfaces

- [Packaged runbook presets](../runbooks/presets): checked-in starter runbooks for common orchestration routes.
- [`dnadesign.ops`](../README.md): lightweight package entrypoint for repo-level routing.

### Boundary reminders

- Ops owns control-plane orchestration, audit trails, and scheduler command ordering.
- Tool-specific runtime semantics stay in the boundary-owning tool docs.
- Durable USR-backed data-plane workflows stay in shared USR operations docs.
