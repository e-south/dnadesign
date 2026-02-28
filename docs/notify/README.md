## Notify Operations

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-28

`notify` is the observer-plane CLI for webhook delivery from local runs and batch workflows.
Use this page as the route map; deep watcher procedures live in `docs/notify/usr-events.md`.

### Choose a workflow

- Send one-off run status messages: use `notify send`.
- Run long-lived Universal Sequence Record watcher loops: use [Notify USR events operator manual](usr-events.md).
- Run watcher loops under BU SCC scheduler patterns: use [BU SCC Batch + Notify runbook](../bu-scc/batch-notify.md).

### Progressive disclosure path

1. Start with the 2-minute path on this page.
2. Use `notify profile doctor` before first live watch to validate profile, event source, webhook source, and TLS settings.
3. Move to [Notify USR events operator manual](usr-events.md) for setup/run/recover command anatomy and failure handling.
4. Use [BU SCC Batch + Notify runbook](../bu-scc/batch-notify.md) only when running under scheduler-managed loops.

### 2-minute operator path

```bash
WORKSPACE=<workspace>

# 1) Provision a reusable secret ref.
WEBHOOK_REF="$(uv run notify setup webhook --secret-source auto --name densegen-shared --json | python -c 'import json,sys; print(json.load(sys.stdin)["webhook"]["ref"])')"

# 2) Create or refresh the workspace profile.
uv run notify setup slack \
  --tool densegen \
  --workspace "$WORKSPACE" \
  --secret-source auto \
  --secret-ref "$WEBHOOK_REF" \
  --policy densegen

# 3) Validate profile wiring before watch.
uv run notify profile doctor --profile src/dnadesign/densegen/workspaces/$WORKSPACE/outputs/notify/densegen/profile.json

# 4) Run the watcher.
uv run notify usr-events watch --tool densegen --workspace "$WORKSPACE" --follow --wait-for-events
```

### Interface contract summary

- Notify consumes Universal Sequence Record `"<dataset>/.events.log"` JSONL only.
- DenseGen runtime telemetry (`outputs/meta/events.jsonl`) is not a Notify input.
- Profile schema contract is `profile_version: 2`.
- Exactly one webhook source is required: `--url`, `--url-env`, or `--secret-ref`.
- Live HTTPS delivery requires trust roots via `--tls-ca-bundle` or `SSL_CERT_FILE`.
- No silent fallback: invalid profile, missing secrets, or invalid event source exits with actionable errors.
- `notify setup slack` mode contract:
  - explicit mode: `--events`
  - resolver mode: `--tool` with exactly one of `--workspace` or `--config`
  - mixed explicit+resolver flags are invalid
- `notify usr-events watch` mode contract:
  - one of `--profile`, `--events`, or resolver mode (`--tool` + exactly one of `--workspace` or `--config`)
  - if resolver-mode auto-profile is missing, watch exits with setup guidance instead of creating implicit state

### Command surface map

| Command | Use when | Output contract |
| --- | --- | --- |
| `notify send` | One-off notifications from scripts or jobs | Immediate post (or formatted payload with `--dry-run`) |
| `notify setup webhook` | Provision or resolve webhook secret refs | Machine-readable `webhook.ref` with `--json` |
| `notify setup slack` | Build workspace-bound watcher profile | Profile JSON + cursor/spool path defaults |
| `notify setup resolve-events` | Verify resolved `.events.log` path without writing profile | Path/policy resolution output |
| `notify profile doctor` | Validate profile wiring and secret resolution | Human-readable status or JSON diagnostics |
| `notify usr-events watch` | Stream `.events.log` to webhook | Restart-safe watch loop with cursor support |
| `notify spool drain` | Retry failed deliveries from spool | Replayed payloads with fail-fast option |

### Maintainer route map

- CLI router/group wiring: `src/dnadesign/notify/cli/__init__.py`.
- Command binding layer:
  - `src/dnadesign/notify/cli/bindings/__init__.py`: binding surface and handler wiring.
  - `src/dnadesign/notify/cli/bindings/deps.py`: dependency exports used by handlers and tests.
  - `src/dnadesign/notify/cli/bindings/registry.py`: Typer command registration wiring.
- Option declarations and command registration: `src/dnadesign/notify/cli/commands/`.
- Command execution handlers:
  - `src/dnadesign/notify/cli/handlers/profile/`: `init_cmd`, `wizard_cmd`, `show_cmd`, `doctor_cmd`
  - `src/dnadesign/notify/cli/handlers/setup/`: `slack_cmd`, `webhook_cmd`, `resolve_events_cmd`, `list_workspaces_cmd`
  - `src/dnadesign/notify/cli/handlers/runtime/`: `watch_cmd`, `spool_cmd`
  - `src/dnadesign/notify/cli/handlers/send.py`
- Runtime and event-processing primitives:
  - `src/dnadesign/notify/runtime/watch_runner.py`: watch runner entrypoint.
  - `src/dnadesign/notify/runtime/watch_runner_contract.py`: watch option contract checks.
  - `src/dnadesign/notify/runtime/watch_runner_resolution.py`: watch source and webhook resolution.
  - `src/dnadesign/notify/runtime/watch_events.py`: event parsing/filtering and payload preparation.
  - `src/dnadesign/notify/runtime/watch_delivery.py`: dry-run output and webhook/spool delivery outcomes.
  - `src/dnadesign/notify/runtime/cursor/`: cursor offset, lock, and follow-loop iteration modules.
  - `src/dnadesign/notify/delivery/secrets/`: secret reference contracts plus keyring/file/shell backend operations.
  - `src/dnadesign/notify/runtime/`, `src/dnadesign/notify/events/`, `src/dnadesign/notify/tool_events/`, `src/dnadesign/notify/delivery/`.

### Troubleshooting and recovery

- `profile doctor` fails on missing webhook env var: confirm `--url-env` target is exported.
- `usr-events watch` reports events-source mismatch: rerun setup with matching `--tool` + `--workspace` and use `--force`.
- HTTPS delivery fails with CA errors: set `--tls-ca-bundle` or export `SSL_CERT_FILE`.
- Repeated delivery failures: run `notify spool drain --profile <profile.json>` after restoring connectivity.

### Canonical runbooks

- Watcher onboarding and lifecycle: [Notify USR events operator manual](usr-events.md).
- BU SCC qsub patterns and notify wiring: [BU SCC Batch + Notify runbook](../bu-scc/batch-notify.md).
- DenseGen submit-ready cluster flow: [BU SCC quickstart](../bu-scc/quickstart.md).
