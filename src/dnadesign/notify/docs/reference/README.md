## Notify reference index

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-16

Use this section when you need strict command, profile, and boundary contracts.

### Read order

1. [Command contracts](command-contracts.md): setup/send/watch/spool/profile invocation rules and failure behavior.
2. [USR event schema reference](../../../usr/docs/reference/event-log.md): upstream event payload contract consumed by Notify.
3. [Maintainer architecture map](../dev/architecture.md): command registration and runtime module ownership.

### Need operator steps instead?

- [Notify USR events runbook](../../../../../docs/notify/usr-events.md): shared setup/watch/recover procedure for day-to-day operator workflows.

### Coverage

- In scope: command-mode exclusivity, webhook source invariants, profile schema/version checks, watcher runtime boundaries.
- Out of scope: BU SCC scheduler policy and cross-tool platform runbooks (use repository `docs/` for those).

### Verify next

- Before first live watcher run: [notify profile doctor](command-contracts.md#notify-profile-doctor).
- Before one-off delivery from scripts: [notify send](command-contracts.md#notify-send).
