## Notify documentation index

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-01

Tool-local docs live here. Cross-tool workflows stay in top-level `docs/`.

### Ownership boundary

- Tool-local (`src/dnadesign/notify/docs/`): command/reference contracts and maintainer internals.
- Top-level (`docs/notify/`): operator route map and shared workflow runbook.

### Progressive disclosure route

1. Choose one task under **Documentation by workflow** and run its first command.
2. Confirm mode/schema rules in [command contracts](reference/command-contracts.md).
3. Open [maintainer architecture map](dev/architecture.md) only when extending internals.

### Audience and prerequisites

- Operators: start with [Notify USR events runbook](../../../../docs/notify/usr-events.md).
- Maintainers: start with [maintainer architecture map](dev/architecture.md).
- Prerequisites: workspace config, USR `.events.log`, and one webhook source (`--url`, `--url-env`, or `--secret-ref`).
- Verify next: [watch command contract](reference/command-contracts.md#notify-usr-events-watch).

### Documentation by workflow

#### Start or refresh a workspace watcher
- [Notify USR events runbook](../../../../docs/notify/usr-events.md): setup, watch, and recovery loop.
- [notify setup slack contract](reference/command-contracts.md#notify-setup-slack): resolver mode versus explicit events mode.
- [notify usr-events watch contract](reference/command-contracts.md#notify-usr-events-watch): mode families and fail-fast checks.

#### Validate profile and event routing
- [notify profile doctor contract](reference/command-contracts.md#notify-profile-doctor): profile, webhook, and event-source checks.
- [Profile schema contract](reference/command-contracts.md#profile-schema-contract): required fields and version invariants.
- [Observer boundary](reference/command-contracts.md#observer-boundary): USR `.events.log` as Notify input stream.

#### Recover delivery failures
- [Recover flow](../../../../docs/notify/usr-events.md#recover-flow): replay sequence.
- [notify spool drain contract](reference/command-contracts.md#notify-spool-drain): replay behavior and fail-fast mode.

#### Send one-off notifications
- [notify send contract](reference/command-contracts.md#notify-send): required flags, webhook source rules, and dry-run behavior.

#### Run cross-tool or cluster workflows
- [DenseGen -> USR -> Notify tutorial](../../densegen/docs/tutorials/demo_usr_notify.md): local cross-tool path.
- [Notify operations route map](../../../../docs/notify/README.md): repository-level operator routing.
- [BU SCC batch + notify runbook](../../../../docs/bu-scc/batch-notify.md): scheduler-oriented workflow.

#### Extend and debug internals
- [Maintainer architecture map](dev/architecture.md): module boundaries and extension seams.
- [Runtime evidence pointers](reference/command-contracts.md#runtime-evidence-pointers): code locations that enforce runtime contracts.

### Documentation by type

- [Operator runbook](../../../../docs/notify/usr-events.md): runnable setup/watch/recover paths.
- [Reference index](reference/README.md): strict command, schema, and boundary contracts.
- [Maintainer internals](dev/architecture.md): architecture map for package extension work.
- [Package entrypoint](../README.md): lightweight tool README for repo-level routing.
- [Repository docs index](../../../../docs/README.md): cross-tool workflow lanes.
