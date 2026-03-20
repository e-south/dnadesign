## Notify documentation index

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-16

Notify tool docs live here. Shared watcher workflows stay in top-level `docs/notify/` or in the package that owns the dataset handoff.
If you need to set up, run, or recover a watcher, start with [Notify USR events runbook](../../../../docs/notify/usr-events.md). Return here for package reference or maintainer docs.

### Ownership boundary

- Tool-local (`src/dnadesign/notify/docs/`): command/reference contracts and maintainer internals.
- Shared operator docs (`docs/notify/` and operations docs such as `src/dnadesign/usr/docs/operations/`): operator guides and cross-tool runbooks.

### Start here

1. If you are operating a watcher, start with [Notify USR events runbook](../../../../docs/notify/usr-events.md); if you are maintaining package-local behavior, continue with **Documentation by workflow** below.
2. Confirm mode/schema rules in [command contracts](reference/command-contracts.md).
3. Open [maintainer architecture map](dev/architecture.md) only when extending internals.

### Audience and prerequisites

- Operators: start with [Notify USR events runbook](../../../../docs/notify/usr-events.md).
- Maintainers: start with [maintainer architecture map](dev/architecture.md).
- Prerequisites: workspace config, USR `.events.log`, and one webhook source (`--url`, `--url-env`, or `--secret-ref`).
- Verify next: [watch command contract](reference/command-contracts.md#notify-usr-events-watch).

### Operator quick path (3 commands)

```bash
uv run notify setup list-workspaces --tool <tool>
uv run notify setup slack --tool <tool> --workspace <workspace-name> --secret-source file --secret-ref file://<path-to-webhook-file>
uv run notify usr-events watch --tool <tool> --workspace <workspace-name> --follow
```

Use [Notify USR events runbook](../../../../docs/notify/usr-events.md) for explicit `--events` mode, recovery, and secret-source alternatives.

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

#### Route to shared cross-tool and scheduler docs
- [DenseGen -> USR -> Notify tutorial](../../densegen/docs/tutorials/demo_usr_notify.md): shared tutorial for one local cross-tool path.
- [BU SCC batch + notify runbook](../../../../docs/bu-scc/batch-notify.md): shared scheduler-oriented workflow.

#### Extend and debug internals
- [Maintainer architecture map](dev/architecture.md): module boundaries and extension seams.
- [Runtime evidence pointers](reference/command-contracts.md#runtime-evidence-pointers): code locations that enforce runtime contracts.

### Documentation by type

- [Operator runbook](../../../../docs/notify/usr-events.md): runnable setup/watch/recover paths.
- [Reference index](reference/README.md): strict command, schema, and boundary contracts.
- [Maintainer internals](dev/architecture.md): architecture map for package extension work.
- [Package entrypoint](../README.md): package README for repo-level routing.
