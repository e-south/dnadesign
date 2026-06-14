## Notify documentation index

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-14

Use this page for Notify command contracts and maintainer docs. If you need to set up, run, or recover a watcher, start with [Notify USR events runbook](../../../../docs/notify/usr-events.md).

### Ownership boundary

- Tool-local (`src/dnadesign/notify/docs/`): command/reference contracts and maintainer internals.
- Shared watcher docs (`docs/notify/` and operations docs such as `src/dnadesign/usr/docs/operations/`): setup, recovery, and cross-tool handoffs.

### Start here

1. Read [command contracts](reference/command-contracts.md) for flags, schemas, and failure rules.
2. Read [maintainer architecture map](dev/architecture.md) when changing package internals.
3. Use [Notify USR events runbook](../../../../docs/notify/usr-events.md) when the task is watcher setup, watching, or recovery.

### Prerequisites

- Users: start with [Notify USR events runbook](../../../../docs/notify/usr-events.md).
- Maintainers: start with [maintainer architecture map](dev/architecture.md).
- Prerequisites: workspace config, USR `.events.log`, and one webhook source (`--url`, `--url-env`, or `--secret-ref`).
- Verify next: [watch command contract](reference/command-contracts.md#notify-usr-events-watch).

### Package docs by task

#### Validate watcher config and event source
- [notify profile doctor contract](reference/command-contracts.md#notify-profile-doctor): profile, webhook, and event-source checks.
- [notify setup slack contract](reference/command-contracts.md#notify-setup-slack): resolver mode versus explicit events mode.
- [notify usr-events watch contract](reference/command-contracts.md#notify-usr-events-watch): mode families and fail-fast checks.
- [Profile schema contract](reference/command-contracts.md#profile-schema-contract): required fields and version invariants.
- [Observer boundary](reference/command-contracts.md#observer-boundary): USR `.events.log` as Notify input stream.

#### Recover delivery failures
- [Recover flow](../../../../docs/notify/usr-events.md#recover-flow): replay sequence.
- [notify spool drain contract](reference/command-contracts.md#notify-spool-drain): replay behavior and fail-fast mode.

#### Send one-off notifications
- [notify send contract](reference/command-contracts.md#notify-send): required flags, webhook source rules, and dry-run behavior.

#### Shared watcher and scheduler docs
- [DenseGen -> USR -> Notify tutorial](../../densegen/docs/tutorials/demo_usr_notify.md): shared tutorial for one local cross-tool path.
- [BU SCC batch + notify runbook](../../../../docs/bu-scc/runbooks/batch-notify.md): shared scheduler-oriented workflow.

#### Extend and debug internals
- [Maintainer architecture map](dev/architecture.md): module boundaries and extension seams.
- [Runtime evidence pointers](reference/command-contracts.md#runtime-evidence-pointers): code locations that enforce runtime contracts.

### Documentation by type

- [Operator runbook](../../../../docs/notify/usr-events.md): runnable setup/watch/recover paths.
- [Reference index](reference/README.md): strict command, schema, and boundary contracts.
- [Maintainer internals](dev/architecture.md): architecture map for package extension work.
- [Package entrypoint](../README.md): package README with the short tool overview.
