![Ops banner](assets/ops-banner.svg)

Ops is the cross-tool orchestration control plane for deterministic batch workflows. It compiles runbook intent into explicit preflight, verification, and submit phases with auditable outputs.

Use Ops when you need repeatable scheduler handoffs across producer and observer tools while preserving workspace-scoped logging and fail-fast contracts. Use shared USR operations docs instead when the durable owner is a data-plane workflow rather than scheduler orchestration.

## Documentation

- [ops docs index](docs/README.md): package-local route map for orchestration docs and packaged runbook assets.
- [Ops orchestration index](../../../docs/operations/README.md): task-first router for runbook lifecycle choices.
- [Orchestration runbooks](../../../docs/operations/orchestration-runbooks.md): runbook schema, command sequence, and contract rules.
- [Packaged runbook presets](runbooks/presets): checked-in starter runbooks for common orchestration routes.
- [Repository docs index](../../../docs/README.md): repo-wide route map for cross-tool workflows.
