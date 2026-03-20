## Ops docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-20

Use this index for package-local Ops documentation. Shared operator procedures stay under top-level `docs/operations/` because Ops owns the repository's control-plane runbook surface, while `ops` itself stays a shared catalog view over `docs/runbooks/README.md` rather than a second registry.

### Start here

- [Runbook catalog](../../../../docs/runbooks/README.md): repo-level inventory when you want a concise list of authoritative procedures first.
- [How to use Ops](how-to-use-ops.md): quick command guide for catalog discovery, procedure inspection, status checks, and manifest scaffolds.
- [Ops orchestration index](../../../../docs/operations/README.md): task-first router for init, plan, execute, and verification flows.
- [Orchestration runbooks](../../../../docs/operations/orchestration-runbooks.md): authoritative runbook schema, command order, and execution contracts.
- [Repository docs index](../../../../docs/README.md): repo-wide route map when the next step is outside Ops.
- Start from `uv run ops catalog list` when you are entering from the shell rather than browsing docs first.

### Package-local surfaces

- [Packaged runbook presets](../runbooks/presets): checked-in starter runbooks for common orchestration routes.
- [`dnadesign.ops`](../README.md): lightweight package entrypoint for repo-level routing.
- [Runbook catalog progress surfaces](../../../../docs/runbooks/README.md#progress-surface-glossary): glossary for registered progress adapters and explicit campaign manifests.

### Boundary reminders

- Ops owns control-plane orchestration, audit trails, and scheduler command ordering.
- `ops progress show` and `ops progress campaign` are read-only; `ops progress scaffold` prints YAML to stdout unless you pass `--out`.
- Tool-specific runtime semantics stay in the boundary-owning tool docs.
- Durable USR-backed data-plane workflows stay in shared USR operations docs.
