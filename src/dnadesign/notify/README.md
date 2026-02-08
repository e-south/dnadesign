# Notify

## At a glance

**Intent:** Consume USR JSONL mutation events and deliver filtered webhook notifications.

**When to use:**
- Monitor DenseGen/USR pipelines through USR events.
- Send milestone or failure notifications to Slack/Discord/generic webhooks.
- Run restart-safe watchers with cursor state.
- Buffer failed deliveries with spool/drain workflows.

**When not to use:**
- Do not point Notify at DenseGen runtime event logs.
- Do not use Notify as a generic log shipper.

**Boundary / contracts:**
- Input contract is USR `<dataset>/.events.log` only.
- Notify does not import USR internals; it consumes JSONL as an external contract.
- Webhook source must be exactly one of `--url`, `--url-env`, or `--secret-ref`.
- Default profile privacy is strict (`include_args=false`, `include_context=false`, `include_raw_event=false`).

**Start here:**
- [Notify USR events operator manual](../../../docs/notify/usr_events.md) (canonical operators runbook)
- [Slack wizard onboarding](../../../docs/notify/usr_events.md#slack-wizard-onboarding-3-minutes)
- [DenseGen -> USR -> Notify demo](../densegen/docs/demo/demo_usr_notify.md) (local end-to-end demo)

## Start here

- Operators manual (watch/spool/drain plus flags): [Notify USR events operator manual](../../../docs/notify/usr_events.md)
- Minimal operator flow: [Minimal operator quickstart](../../../docs/notify/usr_events.md#minimal-operator-quickstart)
- Secure endpoint setup + deployed pressure test flow: [Secure webhook setup](../../../docs/notify/usr_events.md#secure-webhook-setup-real-endpoints)
- Wizard onboarding flow: [Slack wizard onboarding](../../../docs/notify/usr_events.md#slack-wizard-onboarding-3-minutes)
- USR event schema (source of truth for fields): [USR event log schema](../usr/README.md#event-log-schema)
- DenseGen end-to-end demo: [DenseGen -> USR -> Notify demo](../densegen/docs/demo/demo_usr_notify.md)

## Key boundary

Notify consumes USR events:
- `<usr_root>/<dataset>/.events.log`

Notify does not read DenseGen runtime events:
- `densegen/.../outputs/meta/events.jsonl`
