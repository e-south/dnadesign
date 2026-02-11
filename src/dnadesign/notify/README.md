# Notify

Notify reads USR mutation events and posts selected events to webhook providers.

## Contents
- [At a glance](#at-a-glance)
- [Fast operator path](#fast-operator-path)
- [Read order](#read-order)
- [Key boundary](#key-boundary)
- [Observer Contract](#observer-contract)

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

Artifact placement default:
- keep Notify artifacts with the run workspace being watched:
  - `outputs/notify/<tool>/profile.json`
  - `outputs/notify/<tool>/cursor`
  - `outputs/notify/<tool>/spool/`

## Fast operator path

```bash
# Validate a saved profile.
uv run notify profile doctor --profile outputs/notify/densegen/profile.json

# Preview payloads without posting.
uv run notify usr-events watch --profile outputs/notify/densegen/profile.json --dry-run

# Run live watcher.
uv run notify usr-events watch --profile outputs/notify/densegen/profile.json --follow

# Retry failed payloads from spool.
uv run notify spool drain --profile outputs/notify/densegen/profile.json
```

## Read order

- Canonical operators runbook: [docs/notify/usr_events.md](../../../docs/notify/usr_events.md)
- Module-local pointer page: [docs/usr_events.md](docs/usr_events.md)
- Command anatomy: [notify setup slack flags and expectations](../../../docs/notify/usr_events.md#command-anatomy-notify-setup-slack)
- Setup onboarding: [Slack setup onboarding](../../../docs/notify/usr_events.md#slack-setup-onboarding-3-minutes)
- End-to-end stack demo: [DenseGen -> USR -> Notify demo](../densegen/docs/demo/demo_usr_notify.md)
- USR event schema source: [USR event log schema](../usr/README.md#event-log-schema)

## Key boundary

Notify reads:
- `<usr_root>/<dataset>/.events.log`

Notify does not read:
- `densegen/.../outputs/meta/events.jsonl`

## Observer Contract

Notify is an observer control plane:
- it does not launch tool runs
- it does not mutate tool CLI args, env, or runtime behavior
- it only resolves where USR events are expected and watches that stream

Tool integration contract:
- `notify setup slack --tool <tool> --config <workspace-config.yaml>` resolves expected USR `.events.log` destination
- `notify setup resolve-events --tool <tool> --config <config.yaml>` resolves events path/policy without writing a profile
- supported resolvers: `densegen`, `infer_evo2`
- profile stores `events_source` metadata (`tool`, `config`) so watcher restarts can re-resolve paths and avoid stale bindings
- unsupported tools fail fast with explicit errors (no implicit tool fallback)

Webhook contract:
- secure mode `--secret-source auto|keychain|secretservice`
- env mode supports `--url-env <ENV_VAR>` and defaults to `NOTIFY_WEBHOOK` when omitted
