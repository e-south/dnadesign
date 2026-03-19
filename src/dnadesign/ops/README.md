![Ops banner](assets/ops-banner.svg)

Ops is the cross-tool orchestration control plane for deterministic batch workflows. It compiles runbook intent into explicit preflight, verification, and submit phases with auditable outputs, and it is the right entrypoint when the durable owner is scheduler orchestration rather than a USR data-plane workflow.

## Documentation

- [ops docs index](docs/README.md): package-local route map for orchestration docs and packaged runbook assets.
- [Runbook catalog](../../../docs/runbooks/README.md): repo-level inventory of authoritative cross-tool procedures and tool-local runbook sources.
- [`uv run ops catalog list`](../../../docs/runbooks/README.md): terminal view of the shared runbook catalog without leaving the shell, including query filters such as `--plane data-plane --query infer`.
- [Ops orchestration index](../../../docs/operations/README.md): task-first router for runbook lifecycle choices.
- [Orchestration runbooks](../../../docs/operations/orchestration-runbooks.md): runbook schema, command sequence, and contract rules.
- [Packaged runbook presets](runbooks/presets): checked-in starter runbooks for common orchestration routes.
- [Repository docs index](../../../docs/README.md): repo-wide route map for cross-tool workflows.
