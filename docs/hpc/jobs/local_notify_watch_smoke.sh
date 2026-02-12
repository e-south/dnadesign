#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run a local qsub-like notify watcher smoke flow without scheduler access.

Usage:
  docs/hpc/jobs/local_notify_watch_smoke.sh --mode <env|profile> [options]

Options:
  --mode <env|profile>            Required. Harness mode.
  --repo-root <path>              dnadesign repo root (default: auto-detected).
  --workdir <path>                Working directory for events/capture/state.
  --poll-interval-seconds <sec>   Watch poll interval (default: 0.2).
  --idle-timeout-seconds <sec>    Watch idle timeout (default: 60).
  --help                          Show this message.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODE=""
REPO_ROOT="$DEFAULT_REPO_ROOT"
WORKDIR=""
POLL_INTERVAL_SECONDS="0.2"
IDLE_TIMEOUT_SECONDS="60"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --repo-root)
      REPO_ROOT="${2:-}"
      shift 2
      ;;
    --workdir)
      WORKDIR="${2:-}"
      shift 2
      ;;
    --poll-interval-seconds)
      POLL_INTERVAL_SECONDS="${2:-}"
      shift 2
      ;;
    --idle-timeout-seconds)
      IDLE_TIMEOUT_SECONDS="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "Missing required --mode <env|profile>." >&2
  exit 2
fi
if [[ "$MODE" != "env" && "$MODE" != "profile" ]]; then
  echo "Unsupported mode: $MODE. Use env or profile." >&2
  exit 2
fi

if [[ ! -d "$REPO_ROOT" || ! -f "$REPO_ROOT/pyproject.toml" ]]; then
  echo "Invalid repo root: $REPO_ROOT" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH." >&2
  exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but was not found on PATH." >&2
  exit 2
fi

PYTHON_BIN="python3"

if [[ -z "$WORKDIR" ]]; then
  WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/notify-watch-smoke.XXXXXX")"
else
  mkdir -p "$WORKDIR"
fi
WORKDIR="$(cd "$WORKDIR" && pwd)"

EVENTS_PATH="$WORKDIR/usr/demo/.events.log"
CAPTURE_DIR="$WORKDIR/captures"
CAPTURE_PATH="$CAPTURE_DIR/requests.jsonl"
PORT_PATH="$CAPTURE_DIR/server.port"
PROFILE_PATH="$WORKDIR/outputs/notify/densegen/profile.json"
CURSOR_PATH="$WORKDIR/outputs/notify/densegen/cursor"
SPOOL_DIR="$WORKDIR/outputs/notify/densegen/spool"

mkdir -p "$(dirname "$EVENTS_PATH")" "$CAPTURE_DIR" "$(dirname "$PROFILE_PATH")"

"$PYTHON_BIN" - "$EVENTS_PATH" <<'PY'
import json
import pathlib
import sys

events_path = pathlib.Path(sys.argv[1])
event = {
    "event_version": 1,
    "timestamp_utc": "2026-02-12T12:00:00+00:00",
    "action": "densegen_health",
    "dataset": {"name": "demo", "root": str(events_path.parent)},
    "args": {"status": "completed"},
    "metrics": {
        "rows_written": 3,
        "densegen": {
            "run_quota": 3,
            "rows_written_session": 3,
            "quota_progress_pct": 100.0,
            "tfbs_total_library": 3,
            "tfbs_unique_used": 2,
            "tfbs_coverage_pct": 66.7,
            "plans_attempted": 3,
            "plans_solved": 2,
            "run_elapsed_seconds": 5.0,
        },
    },
    "artifacts": {"overlay": {"namespace": "densegen"}},
    "fingerprint": {"rows": 1, "cols": 2, "size_bytes": 128, "sha256": None},
    "registry_hash": "abc123",
    "actor": {"tool": "densegen", "run_id": "smoke", "host": "localhost", "pid": 1},
    "version": "0.1.0",
}
events_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
PY

"$PYTHON_BIN" - "$CAPTURE_PATH" "$PORT_PATH" <<'PY' &
import pathlib
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

capture_path = pathlib.Path(sys.argv[1])
port_path = pathlib.Path(sys.argv[2])


class CaptureHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        with capture_path.open("a", encoding="utf-8") as handle:
            handle.write(body + "\n")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, _format: str, *_args) -> None:
        return


server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
port_path.write_text(str(server.server_port), encoding="utf-8")
server.serve_forever()
PY
SERVER_PID=$!

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for _ in $(seq 1 200); do
  if [[ -s "$PORT_PATH" ]]; then
    break
  fi
  sleep 0.05
done
if [[ ! -s "$PORT_PATH" ]]; then
  echo "Capture server failed to publish port." >&2
  exit 1
fi

PORT="$(cat "$PORT_PATH")"
export NOTIFY_WEBHOOK="http://127.0.0.1:${PORT}/webhook"

QSUB_SCRIPT="$REPO_ROOT/docs/hpc/jobs/bu_scc_notify_watch.qsub"
if [[ ! -x "$QSUB_SCRIPT" ]]; then
  echo "Missing qsub watcher script: $QSUB_SCRIPT" >&2
  exit 1
fi

if [[ "$MODE" == "profile" ]]; then
  (
    cd "$REPO_ROOT"
    uv run notify setup slack \
      --events "$EVENTS_PATH" \
      --profile "$PROFILE_PATH" \
      --policy densegen \
      --secret-source env \
      --url-env NOTIFY_WEBHOOK \
      --force
  )

  (
    cd "$REPO_ROOT"
    NOTIFY_PROFILE="$PROFILE_PATH" \
    NOTIFY_IDLE_TIMEOUT_SECONDS="$IDLE_TIMEOUT_SECONDS" \
    NOTIFY_POLL_INTERVAL_SECONDS="$POLL_INTERVAL_SECONDS" \
    bash "$QSUB_SCRIPT"
  )
else
  (
    cd "$REPO_ROOT"
    EVENTS_PATH="$EVENTS_PATH" \
    CURSOR_PATH="$CURSOR_PATH" \
    SPOOL_DIR="$SPOOL_DIR" \
    NOTIFY_POLICY="densegen" \
    NOTIFY_NAMESPACE="densegen" \
    WEBHOOK_ENV="NOTIFY_WEBHOOK" \
    NOTIFY_IDLE_TIMEOUT_SECONDS="$IDLE_TIMEOUT_SECONDS" \
    NOTIFY_POLL_INTERVAL_SECONDS="$POLL_INTERVAL_SECONDS" \
    bash "$QSUB_SCRIPT"
  )
fi

CAPTURE_COUNT="$("$PYTHON_BIN" - "$CAPTURE_PATH" <<'PY'
import json
import pathlib
import sys

capture_path = pathlib.Path(sys.argv[1])
if not capture_path.exists():
    raise SystemExit("Capture file is missing.")
lines = [line for line in capture_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not lines:
    raise SystemExit("No webhook requests were captured.")
payloads = [json.loads(line) for line in lines]
if not any("SUCCESS" in str(payload.get("text", "")) for payload in payloads):
    raise SystemExit("Captured payloads did not include a success notification.")
print(len(payloads))
PY
)"

echo "Local notify watcher smoke succeeded."
echo "mode=$MODE"
echo "workdir=$WORKDIR"
echo "events=$EVENTS_PATH"
echo "capture=$CAPTURE_PATH"
echo "requests=$CAPTURE_COUNT"
