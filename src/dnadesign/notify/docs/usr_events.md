# Notify: consuming USR events

This module-local page is a short index.
For full operator procedures, use the canonical runbook:
- [docs/notify/usr_events.md](../../../../docs/notify/usr_events.md)

## Contents
- [Fast path](#fast-path)
- [Related stack docs](#related-stack-docs)
- [Boundary reminder](#boundary-reminder)

## Fast path

```bash
# Validate profile fields and secret wiring.
uv run notify profile doctor --profile outputs/notify.profile.json

# Preview payloads first.
uv run notify usr-events watch --profile outputs/notify.profile.json --dry-run

# Run live.
uv run notify usr-events watch --profile outputs/notify.profile.json --follow
```

## Related stack docs

- DenseGen local end-to-end demo: [../../densegen/docs/demo/demo_usr_notify.md](../../densegen/docs/demo/demo_usr_notify.md)
- DenseGen event-boundary contract: [../../densegen/docs/reference/outputs.md#event-streams-and-consumers-densegen-vs-usr](../../densegen/docs/reference/outputs.md#event-streams-and-consumers-densegen-vs-usr)
- USR event schema: [../../usr/README.md#event-log-schema](../../usr/README.md#event-log-schema)

## Boundary reminder

Notify consumes USR `<dataset>/.events.log` only.
DenseGen `outputs/meta/events.jsonl` is runtime telemetry, not Notify input.
