## DenseGen to USR to Notify tutorial

This tutorial shows the full event-driven operator path from DenseGen generation to USR mutation events to Notify webhook delivery. Read it when you need to verify real watcher behavior end-to-end and avoid mixing DenseGen diagnostics with USR event streams.

### What this tutorial demonstrates
This section defines the core workflow outcomes for this integration tutorial.

- Running DenseGen in USR output mode.
- Locating the resolved USR `.events.log` path from workspace outputs.
- Creating a Notify profile from DenseGen config.
- Validating webhook delivery with a local receiver.

### Prerequisites
This section ensures the runtime stack is available before wiring watchers.

```bash
# Install locked Python dependencies.
uv sync --locked

# Confirm DenseGen CLI is available.
uv run dense --help

# Confirm Notify CLI is available.
uv run notify --help

# Confirm packaged TFBS baseline config validates with solver probe.
uv run dense validate-config --probe-solver -c src/dnadesign/densegen/workspaces/demo_tfbs_baseline/config.yaml
```

### Key config knobs
This section highlights the keys that determine where events are written and how watchers resolve them.

- `densegen.output.targets`: Must include `usr` for Notify workflows.
- `densegen.output.usr.root`: Defines dataset root under workspace outputs.
- `densegen.output.usr.dataset`: Defines dataset id and output namespace.
- `densegen.output.usr.health_event_interval_seconds`: Controls heartbeat event cadence.
- `densegen.run.root`: Anchors workspace-relative output paths.
- `plots.source`: Matters when notebook/plot source selection occurs in multi-sink runs.

### Walkthrough
This section runs a local end-to-end validation with two terminals.

#### 1) Start a local webhook receiver (terminal A)
This step gives you a local endpoint so Notify delivery can be tested without external services.

```bash
# Start a local HTTP receiver that prints webhook payloads.
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

#### 2) Create a USR-mode DenseGen workspace (terminal B)
This step stages a workspace that writes dataset mutation events for Notify to consume.

```bash
# Create workspace from TFBS baseline in USR output mode.
uv run dense workspace init --id usr_notify_trial --from-workspace demo_tfbs_baseline --copy-inputs --output-mode usr

# Enter the workspace.
cd src/dnadesign/densegen/workspaces/usr_notify_trial

# Cache config path for subsequent commands.
CONFIG="$PWD/config.yaml"
```

#### 3) Run DenseGen and inspect event paths
This step executes the run and prints the USR event path that Notify must consume.

```bash
# Run generation from a fresh output state.
uv run dense run --fresh --no-plot -c "$CONFIG"

# Print DenseGen diagnostics path (for runtime debugging only).
ls -la outputs/meta/events.jsonl

# Print canonical USR events path used by Notify.
uv run dense inspect run --usr-events-path -c "$CONFIG"
```

#### 4) Configure Notify profile from DenseGen config
This step creates a profile without manual path guessing and binds it to your local webhook endpoint.

```bash
# Export local webhook URL for env-backed secret resolution.
export DENSEGEN_WEBHOOK="http://127.0.0.1:8787/webhook"

# Create Notify profile using DenseGen config resolution.
uv run notify setup slack --tool densegen --config "$CONFIG" --secret-source env --url-env DENSEGEN_WEBHOOK --policy densegen

# Cache profile path for doctor/watch commands.
PROFILE="outputs/notify/densegen/profile.json"
```

#### 5) Validate and watch events
This step verifies profile correctness, previews payloads, then starts live delivery.

```bash
# Validate profile and event-source resolution.
uv run notify profile doctor --profile "$PROFILE"

# Preview payloads without advancing cursor.
uv run notify usr-events watch --profile "$PROFILE" --dry-run --no-advance-cursor-on-dry-run

# Start live watcher and wait for events.
uv run notify usr-events watch --profile "$PROFILE" --follow --wait-for-events --stop-on-terminal-status --idle-timeout 900
```

#### 6) Emit additional events on demand
This step proves live watcher behavior by generating new events after watcher startup.

```bash
# Resume run and extend quota to generate additional USR mutation events.
uv run dense run --resume --extend-quota 2 --no-plot -c "$CONFIG"
```

### Expected outputs
This section lists the artifacts that confirm each boundary in the flow.

- DenseGen diagnostics: `outputs/meta/events.jsonl`
- USR dataset table: `outputs/usr_datasets/<dataset>/records.parquet`
- USR event stream: `outputs/usr_datasets/<dataset>/.events.log`
- Notify profile: `outputs/notify/densegen/profile.json`
- Notify cursor and spool: `outputs/notify/densegen/cursor`, `outputs/notify/densegen/spool/`

### Troubleshooting
This section captures the common watcher integration footguns.

- No webhook posts appear: confirm terminal A receiver is running and `DENSEGEN_WEBHOOK` is exported.
- `notify profile doctor` fails: recreate profile with `notify setup slack --force ...`.
- Watcher sees no events: rerun `uv run dense inspect run --usr-events-path -c "$CONFIG"` and verify profile points to that file.
- Confusing event sources: use **[observability and events](../concepts/observability_and_events.md)** as the canonical boundary definition.
