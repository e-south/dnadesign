![Notify banner](assets/notify-banner.svg)

Notify watches Universal Sequence Record events and sends webhook notifications with strict fail-fast contracts.
Use it when you need reliable status delivery from local workspaces or scheduler-managed runs without adding hidden runtime state or fallback behavior.
For day-to-day operator workflows, start with the [Notify USR events runbook](../../../docs/notify/usr-events.md).

For cross-tool routing, start at the [repository docs index](../../../docs/README.md).

## Start here in 3 commands

```bash
uv run notify setup list-workspaces --tool <tool>
uv run notify setup slack --tool <tool> --workspace <workspace-name> --secret-source file --secret-ref file://<path-to-webhook-file>
uv run notify usr-events watch --tool <tool> --workspace <workspace-name> --follow
```

Use [Notify USR events runbook](../../../docs/notify/usr-events.md) for explicit `--events` mode, recovery, and secret-source alternatives.

## Documentation map

Read in this order:

1. [Notify USR events runbook](../../../docs/notify/usr-events.md): first stop for day-to-day operator setup, watch, and recovery commands.
2. [Notify operations route](../../../docs/notify/README.md): repository-level operator router for local watchers, recovery, and cross-tool operations.
3. [Notify docs index](docs/README.md): package-local task router for tool workflows, references, and maintainer internals.
4. [Reference index](docs/reference/README.md): strict command, profile, and boundary contracts to confirm behavior before automation.
5. [Maintainer architecture map](docs/dev/architecture.md): package module map and extension seams for command/runtime changes.
6. [BU SCC batch + notify](../../../docs/bu-scc/batch-notify.md): scheduler-oriented execution path for cluster submission and verification.
7. [Repository docs index](../../../docs/README.md): cross-tool workflow routes that connect DenseGen, USR, Infer, and Notify.

## Entrypoint contract

1. Audience: Notify operators and maintainers working in this package.
2. Prerequisites: workspace config and one webhook source (`--url`, `--url-env`, or `--secret-ref`).
3. Verify next: [watch command contract](docs/reference/command-contracts.md#notify-usr-events-watch).

## Boundary reminder

Notify consumes Universal Sequence Record `<dataset>/.events.log` and does not consume DenseGen runtime telemetry (`outputs/meta/events.jsonl`).
