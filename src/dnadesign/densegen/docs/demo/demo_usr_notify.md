## Demo: DenseGen -> USR -> Notify (local end-to-end)

This walkthrough shows the full stack in one local workflow:

- DenseGen generates sequences
- DenseGen writes records plus `densegen` overlays into USR
- Notify reads USR `.events.log` and posts events to a webhook

If this is your first DenseGen run, start with:
- [demo_tfbs_baseline.md](demo_tfbs_baseline.md)

### What this demo teaches

- where each stack component reads and writes data
- which event log Notify should consume (and which one it should not)
- how to verify local end-to-end event delivery

Subprocess boundary for this demo:

1. DenseGen runtime pipeline: Stage-A/Stage-B/solve writes runtime artifacts + USR updates
2. USR event stream: `.events.log` is the canonical mutation feed
3. Notify watcher: consumes USR `.events.log` and sends webhook payloads

### Contents

- [What you will have at the end](#what-you-will-have-at-the-end)
- [Prerequisites](#prerequisites)
- [0) Terminal A: start a tiny local webhook receiver](#0-terminal-a-start-a-tiny-local-webhook-receiver)
- [1) Stage a DenseGen workspace](#1-stage-a-densegen-workspace)
- [2) Confirm USR output wiring (run-scoped root under outputs)](#2-confirm-usr-output-wiring-run-scoped-root-under-outputs)
- [3) Edit `config.yaml` dataset naming (optional but recommended)](#3-edit-configyaml-dataset-naming-optional-but-recommended)
- [4) Validate and run](#4-validate-and-run)
- [5) Inspect the USR dataset and the two event logs](#5-inspect-the-usr-dataset-and-the-two-event-logs)
- [6) Terminal B: run Notify against the USR event log](#6-terminal-b-run-notify-against-the-usr-event-log)
- [Where to go next](#where-to-go-next)

---

### What you will have at the end

Inside your DenseGen workspace:

- DenseGen artifacts:
  - `outputs/tables/records.parquet`
  - `outputs/meta/events.jsonl` (DenseGen runtime telemetry)
- USR dataset (written by DenseGen):
  - `outputs/usr_datasets/<dataset>/records.parquet`
  - `outputs/usr_datasets/<dataset>/_derived/densegen/part-*.parquet`
  - `outputs/usr_datasets/<dataset>/.events.log` (Notify input)

---

## Prerequisites

Run from repo root:

```bash
# Install dependencies from the lockfile.
uv sync --locked
```

You also need a solver backend (CBC or GUROBI, depending on your setup):

```bash
# Validate demo config and probe solver availability.
uv run dense validate-config --probe-solver -c src/dnadesign/densegen/workspaces/demo_tfbs_baseline/config.yaml
```

---

## 0) Terminal A: start a tiny local webhook receiver

This small local server prints incoming webhook POSTs.

```bash
# Start a local HTTP server that prints webhook payloads.
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

In a second terminal, set webhook env var:

```bash
# Point Notify to the local receiver.
export DENSEGEN_WEBHOOK="http://127.0.0.1:8787/webhook"
```

---

## 1) Stage a DenseGen workspace

```bash
# Create a new workspace from the binding-sites demo template.
uv run dense workspace init \
  --id usr_notify_trial \
  --from-workspace demo_tfbs_baseline \
  --copy-inputs \
  --output-mode usr

# Move into the workspace and store config path.
cd src/dnadesign/densegen/workspaces/usr_notify_trial
CONFIG="$PWD/config.yaml"
```

---

## 2) Confirm USR output wiring (run-scoped root under outputs)

DenseGen enforces output roots under workspace `outputs/`.

For this demo, keep USR output inside the workspace. `workspace init --output-mode usr`
already sets:

- `output.targets: [usr]`
- `output.usr.root: outputs/usr_datasets`
- `outputs/usr_datasets/registry.yaml` (seeded when a registry seed file is available)

---

## 3) Edit `config.yaml` dataset naming (optional but recommended)

In `config.yaml`, set a namespaced dataset id:

```yaml
densegen:
  output:
    usr:
      dataset: densegen/usr_notify_trial
```

Why this helps:

- namespacing reduces collision risk across runs
- it keeps USR datasets easier to browse and reason about

`registry.yaml` must define the `densegen` namespace columns used by this sink.

---

## 4) Validate and run

```bash
# Validate config + solver from this workspace.
uv run dense validate-config --probe-solver -c "$CONFIG"

# Run generation from a clean outputs state.
uv run dense run --fresh --no-plot -c "$CONFIG"
```

---

## 5) Inspect the USR dataset and the two event logs

DenseGen runtime telemetry:

```bash
# DenseGen runtime event stream (for DenseGen diagnostics).
ls -la outputs/meta/events.jsonl
```

USR mutation events (Notify input):

```bash
# Print the exact USR .events.log path for this run.
uv run dense inspect run --usr-events-path -c "$CONFIG"
```

Use only the USR `.events.log` path for Notify.
`outputs/meta/events.jsonl` is DenseGen runtime telemetry and is not Notify input.

```bash
# Derive dataset path directly from the resolved events file (no path guessing).
EVENTS_PATH="$(uv run dense inspect run --usr-events-path -c "$CONFIG")"
DATASET_PATH="$(dirname "$EVENTS_PATH")"
echo "$EVENTS_PATH"
echo "$DATASET_PATH"
```

Optional USR inspection:

```bash
# Show dataset summary.
uv run usr info "$DATASET_PATH"

# Show a few key fields.
uv run usr head "$DATASET_PATH" -n 3 --columns id,sequence

# Tail USR events directly.
uv run usr events tail "$DATASET_PATH" --follow --format json
```

---

## 6) Terminal B: configure Notify from workspace config and watch events

Use setup resolver mode so you do not have to manually copy the events path:
flag-by-flag rationale and profile expectations are documented in
[Notify command anatomy](../../../../../docs/notify/usr-events.md#command-anatomy-notify-setup-slack).

```bash
# Workspace-scoped DenseGen config for this run (not a repo-root config path).
echo "$CONFIG"

# Create Notify profile from DenseGen config.
uv run notify setup slack \
  --tool densegen \
  --config "$CONFIG" \
  --secret-source env \
  --url-env DENSEGEN_WEBHOOK \
  --policy densegen \
  --progress-step-pct 25 \
  --progress-min-seconds 60

# Default profile path for resolver mode.
PROFILE="outputs/notify/densegen/profile.json"
```

For existing runs, `--events` is also valid if you want direct path mode:

```bash
EVENTS_PATH="$(uv run dense inspect run --usr-events-path -c "$CONFIG")"
uv run notify setup slack \
  --events "$EVENTS_PATH" \
  --profile "$PROFILE" \
  --secret-source env \
  --url-env DENSEGEN_WEBHOOK \
  --policy densegen
```

```bash
# Validate profile.
uv run notify profile doctor --profile "$PROFILE"

# Preview payloads without sending (no cursor advance).
uv run notify usr-events watch --profile "$PROFILE" --dry-run --no-advance-cursor-on-dry-run

# Run live watcher.
uv run notify usr-events watch --profile "$PROFILE" --follow --wait-for-events --stop-on-terminal-status --idle-timeout 900
```

You should now see webhook POST payloads in Terminal A.

To validate live updates (not just replay), keep watcher running and trigger new events:

```bash
# Resume generation to emit additional DenseGen-origin USR events.
uv run dense run --resume --no-plot -c "$CONFIG"
```

For deployed endpoints and secret-safe setup (`.env.local`, `--url-env`), see:
- [Notify operators manual](../../../../../docs/notify/usr-events.md)

---

## Where to go next

- DenseGen output contracts: [../reference/outputs.md](../reference/outputs.md)
- USR overlay semantics: [../../../usr/README.md](../../../usr/README.md)
- Notify operators manual: [../../../../../docs/notify/usr-events.md](../../../../../docs/notify/usr-events.md)

---

@e-south
