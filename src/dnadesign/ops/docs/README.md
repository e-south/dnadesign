## ops docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-19

Use this index for package-local Ops documentation. Shared operator procedures stay under top-level `docs/operations/` because Ops owns the control-plane runbook surface for the repository.
When you use `ops` for repo-wide discovery, it is a read-only lens over the shared catalog in `docs/runbooks/README.md`, not a second registry.

### Start here

- [Runbook catalog](../../../../docs/runbooks/README.md): repo-level inventory when you want a concise list of authoritative procedures first.
- `uv run ops catalog list`: terminal inventory for the same shared runbook catalog when you are already in the shell, with filters such as `--plane data-plane --query infer`, `--section tool-sources`, `--section tool-sources --related-to <registry-id>` for typed related tool docs, and `--related-to <registry-id>` for typed related procedures.
- `uv run ops catalog show <registry-id>`: single registered procedure view with owner docs, typed related tool docs, exact deep docs when declared, typed relation detail, required progress inputs, and next shell commands for progress interrogation.
- `uv run ops progress show <registry-id> ...`: read-only progress summary for one registered procedure when you can point at the owner-local artifact inputs.
- `uv run ops progress scaffold <registry-id> ...`: explicit manifest skeleton for one or more registered procedures, with required placeholder fields derived from the shared catalog contract.
- `uv run ops progress scaffold --related-to <registry-id>`: relation-based manifest starting point for one registered procedure plus its typed related procedures.
- `uv run ops progress campaign --manifest <manifest.yaml>`: explicit multi-step campaign summary that preserves owner-local status ownership.
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
