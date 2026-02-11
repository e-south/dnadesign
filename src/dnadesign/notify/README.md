# Notify

Notify reads USR mutation events and posts selected events to webhook providers.

## Contents
- [At a glance](#at-a-glance)
- [Fast operator path](#fast-operator-path)
- [Read order](#read-order)
- [Key boundary](#key-boundary)

## At a glance

Use Notify when you want:
- restart-safe event watching with cursor offsets
- action/tool filtering before delivery
- spool-and-drain behavior for unstable network environments

Do not use Notify for:
- DenseGen runtime telemetry (`outputs/meta/events.jsonl`)
- generic log shipping

Contract:
- input stream is USR `<dataset>/.events.log` JSONL
- Notify treats USR events as an external, versioned contract
- without `--profile`, choose exactly one webhook source:
  - `--url`
  - `--url-env`
  - `--secret-ref`

Default profile privacy is strict:
- `include_args=false`
- `include_context=false`
- `include_raw_event=false`

## Fast operator path

```bash
# Validate a saved profile.
uv run notify profile doctor --profile outputs/notify.profile.json

# Preview payloads without posting.
uv run notify usr-events watch --profile outputs/notify.profile.json --dry-run

# Run live watcher.
uv run notify usr-events watch --profile outputs/notify.profile.json --follow

# Retry failed payloads from spool.
uv run notify spool drain --profile outputs/notify.profile.json
```

## Read order

- Canonical operators runbook: [docs/notify/usr_events.md](../../../docs/notify/usr_events.md)
- Module-local pointer page: [docs/usr_events.md](docs/usr_events.md)
- Wizard onboarding: [Slack wizard onboarding](../../../docs/notify/usr_events.md#slack-wizard-onboarding-3-minutes)
- End-to-end stack demo: [DenseGen -> USR -> Notify demo](../densegen/docs/demo/demo_usr_notify.md)
- USR event schema source: [USR event log schema](../usr/README.md#event-log-schema)

## Key boundary

Notify reads:
- `<usr_root>/<dataset>/.events.log`

Notify does not read:
- `densegen/.../outputs/meta/events.jsonl`
