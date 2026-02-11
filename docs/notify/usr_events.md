# Notify: consuming USR events

Notify consumes USR mutation events from `.events.log` JSONL files and sends selected events to webhook providers. The integration contract is USR `.events.log` only; DenseGen runtime diagnostics (`outputs/meta/events.jsonl`) are out of scope for Notify.

## Contents
- [Minimal operator quickstart](#minimal-operator-quickstart)
- [Slack wizard onboarding (3 minutes)](#slack-wizard-onboarding-3-minutes)
- [Common mistakes](#common-mistakes)
- [Required event fields (minimum contract)](#required-event-fields-minimum-contract)
- [Quickstart: watch an event log and POST to a webhook](#quickstart-watch-an-event-log-and-post-to-a-webhook)
- [Secure webhook setup (real endpoints)](#secure-webhook-setup-real-endpoints)
- [Profile setup notes](#profile-setup-notes)
- [Deployed pressure test: DenseGen USR Notify](#deployed-pressure-test-densegen-usr-notify)
- [Filtering (reduce noise first, then tighten)](#filtering-reduce-noise-first-then-tighten)
- [Spool and drain (HPC-friendly)](#spool-and-drain-hpc-friendly)

See also:
- USR event schema: [USR README: event log schema](../../src/dnadesign/usr/README.md#event-log-schema)
- DenseGen end-to-end demo: [DenseGen -> USR -> Notify demo](../../src/dnadesign/densegen/docs/demo/demo_usr_notify.md)

---

## Minimal operator quickstart

```bash
# Validate profile and secret resolution.
uv run notify profile doctor --profile outputs/notify.profile.json

# Preview payloads without posting.
uv run notify usr-events watch --profile outputs/notify.profile.json --dry-run

# Run live watcher.
uv run notify usr-events watch --profile outputs/notify.profile.json --follow

# Retry failed payloads from spool.
uv run notify spool drain --profile outputs/notify.profile.json
```

---

## Slack wizard onboarding (3 minutes)

Run these from your DenseGen workspace:

```bash
# 1) Print the exact USR events path
uv run dense inspect run --usr-events-path
```

If you are not in the DenseGen workspace directory, include config explicitly:

```bash
uv run dense inspect run --usr-events-path -c /abs/path/to/config.yaml
```

Concrete example for the three-TF demo workspace (run from repo root):

```bash
# Create a workspace before resolving events path from config.
uv run dense workspace init --id meme_three_tfs_trial --from-workspace demo_meme_three_tfs --copy-inputs --output-mode usr

# Resolve USR events path directly from that workspace config.
EVENTS_PATH="$(uv run dense inspect run --usr-events-path -c src/dnadesign/densegen/workspaces/meme_three_tfs_trial/config.yaml)"
```

Important: `--events` for notify must be a USR `.events.log` JSONL path, not `config.yaml`.
If `--usr-events-path` fails with `output.targets must include 'usr'`, the selected config is not writing USR outputs.

```bash
# 2) Create Slack profile (secure backend mode; URL prompt is hidden)
uv run notify profile wizard \
  --profile outputs/notify.profile.json \
  --provider slack \
  --events <PASTE_EVENTS_PATH_FROM_ABOVE> \
  --cursor outputs/notify.cursor \
  --spool-dir outputs/notify_spool \
  --secret-source auto \
  --preset densegen
```

```bash
# 3) Validate and preview, then run live
uv run notify profile doctor --profile outputs/notify.profile.json
uv run notify usr-events watch --profile outputs/notify.profile.json --dry-run
uv run notify usr-events watch --profile outputs/notify.profile.json --follow
```

If `--secret-source auto` fails (no keychain/secretservice backend), use env mode:

```bash
# Capture webhook URL without shell echo.
read -rsp "Slack Webhook URL: " DENSEGEN_WEBHOOK; echo

# Export for this shell session.
export DENSEGEN_WEBHOOK

# Create profile using env-backed secret mode.
uv run notify profile wizard \
  --profile outputs/notify.profile.json \
  --provider slack \
  --events <PASTE_EVENTS_PATH_FROM_ABOVE> \
  --cursor outputs/notify.cursor \
  --spool-dir outputs/notify_spool \
  --secret-source env \
  --url-env DENSEGEN_WEBHOOK \
  --preset densegen
```

---

## Common mistakes

1) Watching the wrong event log:
- Correct: USR `<dataset>/.events.log`
- Incorrect: DenseGen `outputs/meta/events.jsonl` or DenseGen `config.yaml`

2) Pointing `--events` to `config.yaml`:
- `--events` must be JSONL, first non-empty line must be a USR event object.

3) Writing profile-relative paths as shell-relative paths:
- `events`, `cursor`, and `spool_dir` resolve relative to the profile file location.

---

## Required event fields (minimum contract)

Each JSONL line must include:

- `event_version` (integer)
- `action` (string)

Common optional fields used by payload mapping:

- `timestamp_utc`
- `dataset`
- `actor` (`tool`, `run_id`, `host`, `pid`)
- `metrics`
- `artifacts`
- `registry_hash`

Unknown `event_version` is rejected by default.
Use `--allow-unknown-version` only during controlled migrations.

---

## Quickstart: watch an event log and POST to a webhook

```bash
# Start a watcher on a concrete .events.log path.
uv run notify usr-events watch \
  --events /path/to/.events.log \
  --cursor /path/to/notify.cursor \
  --provider generic \
  --url https://example.com/webhook
```

Behavior:

- reads to EOF in batch mode by default
- use `--follow` to tail continuously
- cursor stores a byte offset for restart-safe resume
- filter by action/tool with `--only-actions` and `--only-tools`
- retries use `--retry-max` and `--retry-base-seconds`
- failed sends can be spooled with `--spool-dir`
- `--dry-run` prints formatted payloads and does not require `--url`, `--url-env`, or `--secret-ref`

---

## Secure webhook setup (real endpoints)

Use `--url-env` or `--secret-ref` for deployed runs so webhook URLs are not committed to repo files.

One-time per shell (no terminal echo):

```bash
# Read webhook URL safely from stdin.
read -rsp "Webhook URL: " DENSEGEN_WEBHOOK; echo

# Export for notify commands.
export DENSEGEN_WEBHOOK
```

Persistent local setup (`.env.local` is a file, not a folder):

```bash
# Write URL into local env file (do not commit this file).
cat > .env.local <<'EOF'
DENSEGEN_WEBHOOK=https://...
EOF
chmod 600 .env.local

# Load values into current shell.
set -a; source .env.local; set +a
```

Security rules:

- Never commit webhook URLs in `config.yaml`, docs, or scripts.
- Prefer provider-scoped endpoints (for example, one Slack webhook per workspace/team).
- Rotate webhooks periodically.
- Minimize payload surface using `--only-actions`, `--only-tools`, and defaults (no args/context/raw event).
- Use `--include-context` and `--include-raw-event` only when actively debugging.

---

## Profile setup notes

Use the Slack wizard quickstart above for onboarding. If you already have a profile:

- `--events`, `--cursor`, and `--spool-dir` are resolved relative to the profile file location.
- If the profile is under `outputs/`, use `usr_datasets/...` rather than `outputs/usr_datasets/...`.
- `--secret-source auto` requires Keychain (macOS) or Secret Service (Linux).
- Use `--secret-source env --url-env DENSEGEN_WEBHOOK` only when you explicitly want env-based secrets.
- If your environment restricts the default state directory, pass explicit writable paths:
  `--cursor outputs/notify.cursor --spool-dir outputs/notify_spool`.

Validate wiring before posting:

```bash
# Validate profile before posting any payload.
uv run notify profile doctor --profile outputs/notify.profile.json
```

Profile security contract:

- Profile files must not contain plaintext `url` values.
- Secrets are resolved via one explicit source:
  - environment variable (`webhook.source=env` in profile v2, or `url_env` in profile v1)
  - secret reference (`webhook.source=secret_ref`, for example keychain/secretservice)
- Unknown profile keys hard-fail to prevent silent config drift.
- Relative profile paths (for example `events`, `cursor`, `spool_dir`) are resolved relative to
  the profile file location, not the current shell directory.
- Profiles default to `include_args=false`, `include_context=false`, and `include_raw_event=false`.

---

## Deployed pressure test: DenseGen USR Notify

Run the watcher in dry-run mode first:

```bash
# Dry-run with action and tool filters to verify payloads first.
uv run notify usr-events watch \
  --events outputs/usr_datasets/demo_pwm/.events.log \
  --cursor outputs/notify.cursor \
  --provider slack \
  --url-env DENSEGEN_WEBHOOK \
  --only-tools densegen \
  --only-actions densegen_health,densegen_flush_failed,materialize \
  --dry-run
```

If payloads look correct, remove `--dry-run` to deliver for real.

Generate additional events by increasing quota and resuming DenseGen:

```bash
# Emit additional USR events by resuming DenseGen.
uv run dense run --resume --no-plot
```

---

## Provider choice: Slack vs email

- Slack (recommended pressure-test path): `--provider slack` with a Slack webhook URL.
- Email: use `--provider generic` to post to your email relay webhook endpoint.
  Notify does not send SMTP directly.

---

## Local test (no dependencies): run a tiny webhook receiver

```bash
# Start local webhook receiver for end-to-end validation.
python - <<'PY'
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(size).decode("utf-8", errors="replace")
        print("\n--- webhook POST ---")
        print(self.path)
        print(body)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok\n")

HTTPServer(("127.0.0.1", 8787), Handler).serve_forever()
PY
```

Then:

```bash
# Point notify at the local receiver.
export WEBHOOK_URL="http://127.0.0.1:8787/webhook"

# Follow events and post to local endpoint.
uv run notify usr-events watch \
  --events /path/to/.events.log \
  --cursor /tmp/notify.cursor \
  --provider generic \
  --url "$WEBHOOK_URL" \
  --follow
```

---

## Filtering (reduce noise first, then tighten)

Two common filters:

- Only these event actions:
  - `--only-actions densegen_health,densegen_flush_failed,materialize`
- Only events from these tools (based on `actor.tool`):
  - `--only-tools densegen`

For DenseGen-in-USR runs, start with:

```bash
--only-tools densegen
--only-actions densegen_health,densegen_flush_failed,materialize
```

Then refine once you have seen real traffic.

---

## Spool and drain (HPC-friendly)

If delivery fails and `--spool-dir` is set, Notify writes payload files locally.
Drain later from a host with stable network access.

```bash
# Watch with spooling enabled and limited retries.
uv run notify usr-events watch \
  --events /path/to/.events.log \
  --cursor /path/to/notify.cursor \
  --provider slack \
  --url-env DENSEGEN_WEBHOOK \
  --spool-dir /path/to/spool \
  --retry-max 2
```

Drain:

```bash
# Retry payloads that were written to spool.
uv run notify spool drain \
  --spool-dir /path/to/spool \
  --provider slack \
  --url-env DENSEGEN_WEBHOOK
```

Successful sends remove spool files. Failed sends are kept for retry.
