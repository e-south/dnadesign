# Notify: consuming Universal Sequence Record events

This module-local page is a compact quick-operations guide.
For full operator procedures, use the canonical runbook:
- [docs/notify/usr-events.md](../../../../docs/notify/usr-events.md)

## Contents
- [Fast path](#fast-path)
- [Related stack docs](#related-stack-docs)
- [Boundary reminder](#boundary-reminder)

## Fast path

```bash
CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml
RUN_ROOT="$(dirname "$CONFIG")"
NOTIFY_DIR="$RUN_ROOT/outputs/notify/densegen"

# Validate profile fields and secret wiring.
uv run notify profile doctor --profile "$NOTIFY_DIR/profile.json"

# Preview payloads first.
uv run notify usr-events watch --profile "$NOTIFY_DIR/profile.json" --dry-run

# Run live.
uv run notify usr-events watch --profile "$NOTIFY_DIR/profile.json" --follow

# Retry failed payloads from spool.
uv run notify spool drain --profile "$NOTIFY_DIR/profile.json"
```

## Related stack docs

- DenseGen local end-to-end demo: [../../densegen/docs/tutorials/demo_usr_notify.md](../../densegen/docs/tutorials/demo_usr_notify.md)
- DenseGen event-boundary contract: [../../densegen/docs/reference/outputs.md#event-streams-and-consumers-densegen-vs-usr](../../densegen/docs/reference/outputs.md#event-streams-and-consumers-densegen-vs-usr)
- Universal Sequence Record event schema: [../../usr/README.md#event-log-schema](../../usr/README.md#event-log-schema)
- Setup command guide: [../../../../docs/notify/usr-events.md#command-anatomy-notify-setup-slack](../../../../docs/notify/usr-events.md#command-anatomy-notify-setup-slack)

## Boundary reminder

Notify consumes Universal Sequence Record `<dataset>/.events.log` only.
DenseGen `outputs/meta/events.jsonl` is runtime telemetry, not Notify input.
