# Notify: consuming Universal Sequence Record events

Notify consumes Universal Sequence Record mutation events from `.events.log` newline-delimited JSON files and sends selected events to webhook providers. The integration contract is Universal Sequence Record `.events.log` only; DenseGen runtime diagnostics (`outputs/meta/events.jsonl`) are out of scope for Notify.

## Contents
- [Minimal operator quickstart](#minimal-operator-quickstart)
- [Artifact placement strategy](#artifact-placement-strategy)
- [Command anatomy: `notify setup slack`](#command-anatomy-notify-setup-slack)
- [Slack setup onboarding (3 minutes)](#slack-setup-onboarding-3-minutes)
- [Common mistakes](#common-mistakes)
- [Required event fields (minimum contract)](#required-event-fields-minimum-contract)
- [Quickstart: watch an event log and send to a webhook](#quickstart-watch-an-event-log-and-send-to-a-webhook)
- [Secure webhook setup (real endpoints)](#secure-webhook-setup-real-endpoints)
- [Profile setup notes](#profile-setup-notes)
- [Deployed pressure test: DenseGen Universal Sequence Record Notify](#deployed-pressure-test-densegen-universal-sequence-record-notify)
- [Filtering (reduce noise first, then tighten)](#filtering-reduce-noise-first-then-tighten)
- [Spool and drain (cluster-friendly)](#spool-and-drain-cluster-friendly)

See also:
- Universal Sequence Record event schema: [Universal Sequence Record README: event log schema](../../src/dnadesign/usr/README.md#event-log-schema)
- DenseGen end-to-end demo: [DenseGen -> Universal Sequence Record -> Notify demo](../../src/dnadesign/densegen/docs/demo/demo_usr_notify.md)

---

## Minimal operator quickstart

```bash
# Run/workspace config for the tool run to watch.
CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml
RUN_ROOT="$(dirname "$CONFIG")"
NOTIFY_DIR="$RUN_ROOT/outputs/notify/densegen"

# Shared computing cluster certificate trust chain for secure webhook delivery.
export SSL_CERT_FILE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem

# Create profile from tool config (observer-only setup).
uv run notify setup slack \
  --tool densegen \
  --config "$CONFIG" \
  --profile "$NOTIFY_DIR/profile.json" \
  --cursor "$NOTIFY_DIR/cursor" \
  --spool-dir "$NOTIFY_DIR/spool" \
  --secret-source auto \
  --policy densegen

# Resolve expected Universal Sequence Record events path from a tool config (no profile write).
uv run notify setup resolve-events --tool densegen --config "$CONFIG"

# Validate profile and secret resolution.
uv run notify profile doctor --profile "$NOTIFY_DIR/profile.json"

# Machine-friendly validation output for automation.
uv run notify profile doctor --profile "$NOTIFY_DIR/profile.json" --json

# Preview payloads without posting.
uv run notify usr-events watch --profile "$NOTIFY_DIR/profile.json" --dry-run

# Run live watcher.
uv run notify usr-events watch --profile "$NOTIFY_DIR/profile.json" --follow

# Tune polling cadence for lower idle overhead on batch nodes.
uv run notify usr-events watch --profile "$NOTIFY_DIR/profile.json" --follow --poll-interval-seconds 1.0

# Retry failed payloads from spool.
uv run notify spool drain --profile "$NOTIFY_DIR/profile.json"
```

---

## Artifact placement strategy

Default (recommended):
- keep Notify runtime artifacts with the run workspace you are watching:
  - `<run_workspace>/outputs/notify/<tool>/profile.json`
  - `<run_workspace>/outputs/notify/<tool>/cursor`
  - `<run_workspace>/outputs/notify/<tool>/spool/`
- this keeps run-local observability state co-located with run-local outputs and avoids cross-run drift
- this remains decoupled from tool sink choices: tools may write parquet and/or Universal Sequence Record outputs, but Notify always watches the Universal Sequence Record `.events.log` contract

Optional centralized mode:
- use a dedicated Notify workspace only when you need shared operator ownership across many runs
- prefer `/project/...` paths
- include tool/run in the watch id, for example:
  - `<dnadesign_repo>/src/dnadesign/notify/workspaces/densegen-<workspace>/outputs/notify/densegen/profile.json`
- keep `events_source` in profile so each watcher remains bound to tool+config contract

Avoid by default:
- writing runtime artifacts under `src/dnadesign/notify/src/...`
- mixing multiple unrelated runs into one cursor/spool path

---

## Command anatomy: `notify setup slack`

Canonical command:

```bash
CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml
RUN_ROOT="$(dirname "$CONFIG")"
NOTIFY_DIR="$RUN_ROOT/outputs/notify/densegen"

uv run notify setup slack \
  --tool densegen \
  --config "$CONFIG" \
  --profile "$NOTIFY_DIR/profile.json" \
  --cursor "$NOTIFY_DIR/cursor" \
  --spool-dir "$NOTIFY_DIR/spool" \
  --secret-source auto \
  --policy densegen
```

What each flag does:
- `--tool densegen`: selects the DenseGen events-path resolver.
- `--config .../config.yaml`: the workspace-scoped DenseGen run config used by the resolver.
- `--profile .../outputs/notify/<tool>/profile.json`: profile JavaScript Object Notation location (run-local by default).
- `--cursor .../outputs/notify/<tool>/cursor`: restart-safe byte offset for watcher resume.
- `--spool-dir .../outputs/notify/<tool>/spool`: durable queue for failed webhook deliveries.
- `--secret-source auto`: selects secure backend (`keychain`/`secretservice`) and prompts for webhook address if needed.
- `--policy densegen`: applies DenseGen defaults for action/tool filters.

Critical expectation for `--config`:
- this is not a repo-root config path
- pass the workspace/run config path, for example:
  - `<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml`
  - `src/dnadesign/densegen/workspaces/<workspace>/config.yaml`
- resolver mode reads output destination settings from this config and computes expected Universal Sequence Record `.events.log`

Profile expectations:
- profile schema is v2 only (`profile_version: 2`)
- profile stores webhook references, not plaintext webhook URLs
- profile stores both resolved `events` and `events_source` (`tool`, `config`) for re-resolution and drift prevention
- profile can point to a future `.events.log` path before the run creates it
- profile file is written with private permissions (`0600`)
- re-running setup on an existing profile path requires `--force`

Optional flags you may need:
- `--url-env <ENV_VAR>`: explicit env-backed secret source (defaults to `NOTIFY_WEBHOOK` when `--secret-source env` and omitted)
- `--secret-ref <backend://service/account>`: explicit key name/location for secure backend
- `--events /abs/path/to/.events.log`: bypass resolver mode and point directly to an existing Universal Sequence Record events file
- `--tls-ca-bundle /path/to/ca-bundle.pem`: explicit certificate-authority bundle for secure webhook delivery
- if `--events` is used with default profile path, pass `--policy` (for namespace) or pass an explicit `--profile`

After setup:

```bash
uv run notify profile doctor --profile "$NOTIFY_DIR/profile.json"
uv run notify usr-events watch --profile "$NOTIFY_DIR/profile.json" --follow --wait-for-events
```

---

## Slack setup onboarding (3 minutes)

Use observer-only setup with DenseGen config resolution.  
`--config` below means your DenseGen `config.yaml` (not a notify file and not a Universal Sequence Record path).

```bash
CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml
RUN_ROOT="$(dirname "$CONFIG")"
NOTIFY_DIR="$RUN_ROOT/outputs/notify/densegen"

# 1) Create a profile from DenseGen config (events path auto-resolved)
uv run notify setup slack \
  --tool densegen \
  --config "$CONFIG" \
  --profile "$NOTIFY_DIR/profile.json" \
  --cursor "$NOTIFY_DIR/cursor" \
  --spool-dir "$NOTIFY_DIR/spool" \
  --secret-source auto \
  --policy densegen
```

Infer/Evo2 config setup follows the same observer contract:

```bash
uv run notify setup slack \
  --tool infer_evo2 \
  --config <dnadesign_repo>/src/dnadesign/infer/workspaces/<workspace>/config.yaml \
  --profile outputs/notify/infer_evo2/profile.json \
  --cursor outputs/notify/infer_evo2/cursor \
  --spool-dir outputs/notify/infer_evo2/spool \
  --secret-source auto \
  --policy infer_evo2
```

The setup command is safe before the run starts:
- it resolves the expected Universal Sequence Record `.events.log` destination from config
- it stores tool/config metadata so watcher runs can re-resolve paths and avoid config-drift
- it does not launch DenseGen or modify DenseGen runtime behavior

Manual Slack webhook address entry via wizard/setup (recommended when secure backend is available):

```bash
# Do not pass --webhook-url. Setup will prompt for it with hidden input.
uv run notify setup slack \
  --tool densegen \
  --config "$CONFIG" \
  --profile "$NOTIFY_DIR/profile.json" \
  --cursor "$NOTIFY_DIR/cursor" \
  --spool-dir "$NOTIFY_DIR/spool" \
  --secret-source auto \
  --policy densegen
```

For existing runs where you already have an events file path:

```bash
uv run notify setup slack \
  --events /abs/path/to/.events.log \
  --profile "$NOTIFY_DIR/profile.json" \
  --cursor "$NOTIFY_DIR/cursor" \
  --spool-dir "$NOTIFY_DIR/spool" \
  --secret-source auto \
  --policy densegen
```

When prompted:
- paste your Slack webhook address
- press Enter (input stays hidden)

For automation/agents, append `--json` to `setup slack`, `setup resolve-events`, and `profile doctor`
to emit structured output with `ok` and error fields.

```bash
# 2) Validate and preview, then run live
# If the run has not started yet, skip dry-run and use --wait-for-events.
uv run notify profile doctor --profile "$NOTIFY_DIR/profile.json"
uv run notify usr-events watch --profile "$NOTIFY_DIR/profile.json" --dry-run
uv run notify usr-events watch --profile "$NOTIFY_DIR/profile.json" --follow
uv run notify usr-events watch --profile "$NOTIFY_DIR/profile.json" --follow --wait-for-events --stop-on-terminal-status --idle-timeout 900
```

`notify profile doctor` treats resolver-backed profiles as valid before `.events.log` is created and reports this as a pending events file state.

```bash
# 3) Run the watcher in Boston University Shared Computing Cluster batch mode (recommended)
qsub -P <project> \
  -v NOTIFY_PROFILE=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/outputs/notify/densegen/profile.json \
  docs/hpc/jobs/bu_scc_notify_watch.qsub
```

If `--secret-source auto` fails (no keychain/secretservice backend), use env mode:

```bash
# Capture webhook address without shell echo.
read -rsp "Slack webhook address: " NOTIFY_WEBHOOK; echo

# Export for this shell session.
export NOTIFY_WEBHOOK

# Create profile using env-backed secret mode.
uv run notify setup slack \
  --tool densegen \
  --config "$CONFIG" \
  --profile "$NOTIFY_DIR/profile.json" \
  --cursor "$NOTIFY_DIR/cursor" \
  --spool-dir "$NOTIFY_DIR/spool" \
  --secret-source env \
  --url-env NOTIFY_WEBHOOK \
  --policy densegen
```

If `--url-env` is omitted with `--secret-source env`, Notify uses `NOTIFY_WEBHOOK` by default.

Policy options:
- `--policy densegen`: DenseGen-focused filters (`densegen_health,densegen_flush_failed,materialize`, tool `densegen`)
- `--policy infer_evo2`: infer/Evo2-focused filters (`attach,materialize`, tool `infer`)
- `--policy generic`: no default filters

For Boston University Shared Computing Cluster batch mode with environment variables (no profile):

```bash
qsub -P <project> \
  -v NOTIFY_TOOL=densegen,NOTIFY_CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml,WEBHOOK_ENV=NOTIFY_WEBHOOK \
  docs/hpc/jobs/bu_scc_notify_watch.qsub
```

If you run env mode with explicit `EVENTS_PATH` instead of resolver mode, set both
`NOTIFY_POLICY` and `NOTIFY_NAMESPACE` so filter defaults and state paths stay deterministic.

---

## Common mistakes

1) Watching the wrong event log:
- Correct: Universal Sequence Record `<dataset>/.events.log`
- Incorrect: DenseGen `outputs/meta/events.jsonl` or DenseGen `config.yaml`

2) Pointing `--events` to `config.yaml`:
- `--events` must be newline-delimited JSON, and the first non-empty line must be a Universal Sequence Record event object.

3) Writing profile-relative paths as shell-relative paths:
- `events`, `cursor`, and `spool_dir` resolve relative to the profile file location.

4) Secure webhook certificate failures on shared computing cluster nodes:
- Symptom: `CERTIFICATE_VERIFY_FAILED`
- Fix: set `SSL_CERT_FILE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem` or pass `--tls-ca-bundle` explicitly.

---

## Required event fields (minimum contract)

Each newline-delimited JSON line must include:

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

## Quickstart: watch an event log and send to a webhook

```bash
# Start a watcher on a concrete .events.log path.
uv run notify usr-events watch \
  --events /path/to/.events.log \
  --cursor /path/to/notify.cursor \
  --provider generic \
  --url https://example.com/webhook \
  --tls-ca-bundle /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
```

Behavior:

- reads to end-of-file in batch mode by default
- use `--follow` to tail continuously
- cursor stores a byte offset for restart-safe resume
- filter by action/tool with `--only-actions` and `--only-tools`
- retries use `--retry-max` and `--retry-base-seconds`
- failed sends can be spooled with `--spool-dir`
- `--dry-run` prints formatted payloads and does not require `--url`, `--url-env`, or `--secret-ref`
- by default, `--dry-run` still advances cursor offsets to prevent replay drift
- use `--no-advance-cursor-on-dry-run` when you need a non-mutating preview

---

## Secure webhook setup (real endpoints)

Use `--url-env` or `--secret-ref` for deployed runs so webhook URLs are not committed to repo files.

One-time per shell (no terminal echo):

```bash
# Read webhook address safely from standard input.
read -rsp "Webhook address: " NOTIFY_WEBHOOK; echo

# Export for notify commands.
export NOTIFY_WEBHOOK

# Shared computing cluster: export certificate bundle for secure delivery.
export SSL_CERT_FILE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
```

Persistent local setup (`.env.local` is a file, not a folder):

```bash
# Write URL into local env file (do not commit this file).
cat > .env.local <<'EOF'
NOTIFY_WEBHOOK=https://...
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
- Use `--secret-source env --url-env NOTIFY_WEBHOOK` only when you explicitly want env-based secrets.
- Keep watcher state run-local by passing explicit paths:
  `--cursor outputs/notify/densegen/cursor --spool-dir outputs/notify/densegen/spool`.

Validate wiring before posting:

```bash
# Validate profile before posting any payload.
uv run notify profile doctor --profile outputs/notify/densegen/profile.json
```

Profile security contract:

- Profile schema is v2 only (`profile_version: 2`).
- Profile files must not contain plaintext `url` values.
- Profile files are created with private file mode (`0600`).
- Secrets are resolved via one explicit source:
  - environment variable (`webhook.source=env`)
  - secret reference (`webhook.source=secret_ref`, for example keychain/secretservice)
- Unknown profile keys hard-fail to prevent silent config drift.
- `events_source` records (`tool`, `config`) so watcher runs can re-resolve event paths.
- Profiles may reference events files that are not created yet; start watcher with `--wait-for-events`.
- Relative profile paths (for example `events`, `cursor`, `spool_dir`) are resolved relative to
  the profile file location, not the current shell directory.
- Profiles default to `include_args=false`, `include_context=false`, and `include_raw_event=false`.

---

## Deployed pressure test: DenseGen Universal Sequence Record Notify

Run the watcher in dry-run mode first:

```bash
# Dry-run with action and tool filters to verify payloads first.
uv run notify usr-events watch \
  --events outputs/usr_datasets/demo_pwm/.events.log \
  --cursor outputs/notify/densegen/cursor \
  --provider slack \
  --url-env NOTIFY_WEBHOOK \
  --only-tools densegen \
  --only-actions densegen_health,densegen_flush_failed,materialize \
  --dry-run
```

If payloads look correct, remove `--dry-run` to deliver for real.

Generate additional events by increasing quota and resuming DenseGen:

```bash
# Emit additional Universal Sequence Record events by resuming DenseGen.
uv run dense run --resume --no-plot
```

---

## Provider choice: Slack vs email

- Slack (recommended pressure-test path): `--provider slack` with a Slack webhook address.
- Email: use `--provider generic` to post to your email relay webhook endpoint.
  Notify does not send Simple Mail Transfer Protocol traffic directly.

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

For DenseGen-in-Universal-Sequence-Record runs, start with:

```bash
--only-tools densegen
--only-actions densegen_health,densegen_flush_failed,materialize
```

Workflow policy defaults (profile wizard / qsub env mode) are equivalent shortcuts:
- `densegen` -> `--only-tools densegen` and `--only-actions densegen_health,densegen_flush_failed,materialize`
- `infer_evo2` -> `--only-tools infer` and `--only-actions attach,materialize`
- `generic` -> no action/tool filter

Then refine once you have seen real traffic.

---

## Spool and drain (cluster-friendly)

If delivery fails and `--spool-dir` is set, Notify writes payload files locally.
Drain later from a host with stable network access.

```bash
# Watch with spooling enabled and limited retries.
uv run notify usr-events watch \
  --events /path/to/.events.log \
  --cursor /path/to/notify.cursor \
  --provider slack \
  --url-env NOTIFY_WEBHOOK \
  --spool-dir /path/to/spool \
  --retry-max 2
```

Drain:

```bash
# Retry payloads that were written to spool.
uv run notify spool drain \
  --spool-dir /path/to/spool \
  --provider slack \
  --url-env NOTIFY_WEBHOOK
```

Successful sends remove spool files. Failed sends are kept for retry.
If `--provider` is omitted, drain uses the provider stored per spool file.
