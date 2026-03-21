## Ops docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-20

Use this page for Ops package docs, packaged presets, and command reference. If you want the full command list, use `docs/runbooks/README.md`.

### Start here

- [Runbook catalog](../../../../docs/runbooks/README.md): repo-level command index when you want the procedure list first.
- [How to use Ops](how-to-use-ops.md): command guide for catalog discovery, procedure inspection, status checks, and manifest scaffolds.
- [Ops orchestration index](../../../../docs/operations/README.md): docs for init, plan, execute, and verification.
- [Orchestration runbooks](../../../../docs/operations/orchestration-runbooks.md): runbook schema, command order, and execution contracts.
- [Repository docs index](../../../../docs/README.md): docs index when the next step is outside Ops.

If you are entering from the shell rather than browsing docs first, start with `uv run ops catalog list --simple`, then open [How to use Ops](how-to-use-ops.md) for the command summary.

### Package-local surfaces

- [Packaged runbook presets](../runbooks/presets): checked-in starter runbooks for common orchestration routes.
- [`dnadesign.ops`](../README.md): package README with the short tool overview.
- [Runbook catalog progress views](../../../../docs/runbooks/README.md#progress-views): glossary for registered progress adapters and explicit campaign manifests.

### Boundary reminders

- Ops owns control-plane orchestration, audit trails, and scheduler command ordering.
- `ops progress show` and `ops progress campaign` are read-only; `ops progress scaffold` prints YAML to stdout unless you pass `--out`.
- Tool-specific runtime semantics stay in the boundary-owning tool docs.
- Durable USR-backed data-plane workflows stay in shared USR operations docs.
