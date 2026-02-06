## Demo: DenseGen -> USR -> Notify (local end-to-end)

This demo connects the whole stack with minimal moving parts:

- DenseGen generates sequences (binding-sites path).
- DenseGen writes results into a USR dataset (base rows plus `densegen` overlay parts).
- Notify tails the USR `.events.log` and posts selected events to a webhook.

If you are new to DenseGen, run the vanilla demo first:
- [demo_binding_sites.md](demo_binding_sites.md)

---

### What you will have at the end

Inside your DenseGen workspace you will see:

- DenseGen artifacts:
  - `outputs/tables/dense_arrays.parquet`
  - `outputs/meta/events.jsonl` (DenseGen runtime events)
- USR dataset (written by DenseGen):
  - `outputs/usr_datasets/<dataset>/records.parquet`
  - `outputs/usr_datasets/<dataset>/_derived/densegen/part-*.parquet`
  - `outputs/usr_datasets/<dataset>/.events.log` (USR mutation events; Notify consumes this)

---

## Prerequisites

From repo root:

```bash
uv sync --locked
```

You also need a supported solver backend (CBC or GUROBI depending on your setup):

```bash
uv run dense validate-config --probe-solver -c src/dnadesign/densegen/workspaces/demo_binding_sites_vanilla/config.yaml
```

---

## 0) Terminal A: start a tiny local webhook receiver

This is a no-dependency server that prints POST bodies.

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

In a second terminal, set:

```bash
export DENSEGEN_WEBHOOK="http://127.0.0.1:8787/webhook"
```

---

## 1) Stage a DenseGen workspace

```bash
uv run dense workspace init \
  --id demo_usr_notify \
  --template-id demo_binding_sites_vanilla \
  --copy-inputs \
  --output-mode usr

cd src/dnadesign/densegen/workspaces/runs/demo_usr_notify
CONFIG="$PWD/config.yaml"
```

---

## 2) Confirm USR output wiring (run-scoped root under outputs)

DenseGen enforces that output roots live under the workspace `outputs/`.
For this demo, keep the USR root inside the workspace. `workspace init --output-mode usr`
already configures:

- `output.targets: [usr]`
- `output.usr.root: outputs/usr_datasets`
- `outputs/usr_datasets/registry.yaml` (seeded when template is available)

---

## 3) Edit `config.yaml` dataset naming (optional but recommended)

In `config.yaml`, set a namespaced dataset id:

```yaml
densegen:
  output:
    usr:
      dataset: densegen/demo_usr_notify
```

Notes:

- `dataset` can be namespaced (`densegen/demo_usr_notify`) to reduce collisions.
- `registry.yaml` must define the `densegen` namespace columns used by the sink.

---

## 4) Validate and run

```bash
uv run dense validate-config --probe-solver -c "$CONFIG"
uv run dense run --fresh --no-plot -c "$CONFIG"
```

---

## 5) Inspect the USR dataset and the two event logs

DenseGen runtime events:

```bash
ls -la outputs/meta/events.jsonl
```

USR mutation events (Notify input):

```bash
ls -la outputs/usr_datasets/densegen/demo_usr_notify/.events.log
```

Optional: inspect the dataset via USR CLI using the dataset path:

```bash
uv run usr info outputs/usr_datasets/densegen/demo_usr_notify
uv run usr head outputs/usr_datasets/densegen/demo_usr_notify -n 3 --columns id,sequence
uv run usr events tail outputs/usr_datasets/densegen/demo_usr_notify --follow --format json
```

---

## 6) Terminal B: run Notify against the USR event log

```bash
# 1) Get the exact USR events path
uv run dense inspect run --usr-events-path -c "$CONFIG"
```

If Terminal B is not in this run workspace, use:

```bash
uv run dense inspect run --usr-events-path -c src/dnadesign/densegen/workspaces/runs/demo_usr_notify/config.yaml
```

`notify profile wizard --events` expects the USR `.events.log` path from the command above, not `config.yaml`.

```bash
# 2) Create profile (Slack format; works with local receiver URL too)
uv run notify profile wizard \
  --profile outputs/notify.profile.json \
  --provider slack \
  --events <PASTE_EVENTS_PATH_FROM_ABOVE> \
  --secret-source env \
  --url-env DENSEGEN_WEBHOOK \
  --only-tools densegen \
  --only-actions densegen_health,densegen_flush_failed,materialize
```

```bash
# 3) Validate and preview once, then run live
uv run notify profile doctor --profile outputs/notify.profile.json
uv run notify usr-events watch --profile outputs/notify.profile.json --dry-run
uv run notify usr-events watch --profile outputs/notify.profile.json --follow
```

You should see webhook POST bodies printing in Terminal A.

If you want to verify real-time updates instead of replaying existing events, keep Terminal B running and trigger new events from a third shell:

```bash
uv run dense run --resume --allow-quota-increase --no-plot -c "$CONFIG"
```

This emits additional `densegen_health` and `materialize` events that should appear immediately in your webhook receiver output.

For real deployed endpoints (Slack/email relay) and secret-safe setup (`.env.local`, `--url-env`),
see `../../../notify/docs/usr_events.md`.

---

## Where to go next

- DenseGen output contracts: `../reference/outputs.md`
- USR concepts plus overlay semantics: `../../../usr/README.md`
- Notify operators manual: `../../../notify/docs/usr_events.md`

---

@e-south
