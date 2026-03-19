![Ops banner](assets/ops-banner.svg)

Ops is the cross-tool orchestration control plane for deterministic batch workflows. It compiles runbook intent into explicit preflight, verification, and submit phases with auditable outputs, and it is the right entrypoint when the durable owner is scheduler orchestration rather than a USR data-plane workflow.
For repo-wide runbook discovery, `ops` exposes a read-only lens over the shared catalog in `docs/runbooks/README.md`; it does not own a second registry.

## Documentation

- [ops docs index](docs/README.md): package-local route map for orchestration docs and packaged runbook assets.
- [Runbook catalog](../../../docs/runbooks/README.md): repo-level inventory of authoritative cross-tool procedures and owner-local tool entrypoints.
- [`uv run ops catalog list`](../../../docs/runbooks/README.md): terminal view of the shared runbook catalog without leaving the shell, including query filters such as `--plane data-plane --query infer`, `--section tool-sources`, `--section tool-sources --related-to <registry-id>` for typed related tool docs, and `--related-to <registry-id>` for typed related procedures.
- [`uv run ops catalog show`](../../../docs/runbooks/README.md): single-procedure view with owner docs, typed related tool docs, exact deep docs when declared, typed relation detail, required progress inputs, and next shell commands for progress interrogation.
- [`uv run ops progress show`](../../../docs/runbooks/README.md#progress-surface-glossary): read-only summary for one registered progress surface when you already know the explicit artifact inputs.
- [`uv run ops progress scaffold`](../../../docs/runbooks/README.md#explicit-campaign-manifest-shape): emit an explicit manifest skeleton from registry ids and the required progress-field stubs.
- [`uv run ops progress scaffold --related-to`](../../../docs/runbooks/README.md#explicit-campaign-manifest-shape): expand one registered procedure into an explicit starting manifest with its typed related procedures.
- [`uv run ops progress campaign`](../../../docs/runbooks/README.md#explicit-campaign-manifest-shape): read-only multi-step summary driven by an explicit manifest, not an inferred global engine.
- [Ops orchestration index](../../../docs/operations/README.md): task-first router for runbook lifecycle choices.
- [Orchestration runbooks](../../../docs/operations/orchestration-runbooks.md): runbook schema, command sequence, and contract rules.
- [Packaged runbook presets](runbooks/presets): checked-in starter runbooks for common orchestration routes.
- [Repository docs index](../../../docs/README.md): repo-wide route map for cross-tool workflows.
