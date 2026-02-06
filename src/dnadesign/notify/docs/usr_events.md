# Notify: consuming USR events

Notify consumes USR mutation events from `.events.log` JSONL files and sends selected events to webhook providers. The integration contract is USR `.events.log` only; DenseGen runtime diagnostics (`outputs/meta/events.jsonl`) are out of scope for Notify.

See also:
- USR event schema: `../../usr/README.md#event-log-schema`
- DenseGen end-to-end demo: `../../densegen/docs/demo/demo_usr_notify.md`

---

## Minimal operator quickstart

```bash
uv run notify profile doctor --profile outputs/notify.profile.json
uv run notify usr-events watch --profile outputs/notify.profile.json --dry-run
uv run notify usr-events watch --profile outputs/notify.profile.json --follow
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

Concrete example for the three-TF demo workspace:

```bash
EVENTS_PATH="$(uv run dense inspect run --usr-events-path -c src/dnadesign/densegen/workspaces/runs/demo_pwm/config.yaml)"
```

Important: `--events` for notify must be a USR `.events.log` JSONL path, not `config.yaml`.
If `--usr-events-path` fails with `output.targets must include 'usr'`, the selected config is not writing USR outputs.

```bash
# 2) Create Slack profile (secure backend mode; URL prompt is hidden)
uv run notify profile wizard \
  --profile outputs/notify.profile.json \
  --provider slack \
  --events <PASTE_EVENTS_PATH_FROM_ABOVE> \
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
read -rsp "Slack Webhook URL: " DENSEGEN_WEBHOOK; echo
export DENSEGEN_WEBHOOK

uv run notify profile wizard \
  --profile outputs/notify.profile.json \
  --provider slack \
  --events <PASTE_EVENTS_PATH_FROM_ABOVE> \
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
read -rsp "Webhook URL: " DENSEGEN_WEBHOOK; echo
export DENSEGEN_WEBHOOK
```

Persistent local setup (`.env.local` is a file, not a folder):

```bash
cat > .env.local <<'EOF'
DENSEGEN_WEBHOOK=https://...
EOF
chmod 600 .env.local
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

Validate wiring before posting:

```bash
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

## Deployed pressure test: DenseGen -> USR -> Notify

Run the watcher in dry-run mode first:

```bash
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
uv run dense run --resume --allow-quota-increase --no-plot
```

---

## Provider choice: Slack vs email

- Slack (recommended pressure-test path): `--provider slack` with a Slack webhook URL.
- Email: use `--provider generic` to post to your email relay webhook endpoint.
  Notify does not send SMTP directly.

---

## Local test (no dependencies): run a tiny webhook receiver

```bash
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
export WEBHOOK_URL="http://127.0.0.1:8787/webhook"
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
uv run notify spool drain \
  --spool-dir /path/to/spool \
  --provider slack \
  --url-env DENSEGEN_WEBHOOK
```

Successful sends remove spool files. Failed sends are kept for retry.
