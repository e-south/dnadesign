#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=""
PORT=""
STRUCTURE_PATH=""
SESSION_ID="chimerax_session"
TITLE=""

usage() {
  cat <<'USAGE'
Usage: chimerax-session-start.sh --structure PATH [--output-dir DIR] [--port PORT] [--session-id ID] [--title TEXT]

Open a visible ChimeraX session, start localhost REST control, and write a
control-session manifest. The script stops after the session-ready pause point.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --structure)
      STRUCTURE_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --session-id)
      SESSION_ID="$2"
      shift 2
      ;;
    --title)
      TITLE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'FAIL: unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$STRUCTURE_PATH" || ! -f "$STRUCTURE_PATH" ]]; then
  printf 'FAIL: --structure must name an existing local file\n' >&2
  exit 1
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$(mktemp -d -t chimerax-structure-review-session)"
fi
mkdir -p "$OUTPUT_DIR"
if [[ -z "$PORT" ]]; then
  PORT="$(python3 - <<'PY'
import socket

sock = socket.socket()
sock.bind(("127.0.0.1", 0))
print(sock.getsockname()[1])
sock.close()
PY
)"
fi
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1024 || PORT > 65535 )); then
  printf 'FAIL: port must be an integer from 1024 to 65535\n' >&2
  exit 1
fi

preflight="$("$SCRIPT_DIR/chimerax-preflight.sh")"
printf '%s\n' "$preflight"
CHIMERAX_BIN_RESOLVED="$(printf '%s\n' "$preflight" | sed -n 's/^PASS: ChimeraX executable: //p' | head -n 1)"
if [[ -z "$CHIMERAX_BIN_RESOLVED" ]]; then
  printf 'FAIL: could not resolve ChimeraX executable from preflight output\n' >&2
  exit 1
fi

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

validate_cxc_text() {
  local label="$1"
  local value="$2"
  if [[ "$value" == *'"'* || "$value" == *';'* || "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    printf 'FAIL: %s must not contain quotes, semicolons, or newlines for ChimeraX startup commands\n' "$label" >&2
    exit 1
  fi
}

START_SCRIPT="$OUTPUT_DIR/start_session.cxc"
CHIMERAX_LOG="$OUTPUT_DIR/chimerax.log"
SESSION_MANIFEST="$OUTPUT_DIR/control_session.yaml"
COMMAND_LOG="$OUTPUT_DIR/session.commands.jsonl"
ABS_STRUCTURE="$(python3 - "$STRUCTURE_PATH" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
validate_cxc_text "--structure path" "$ABS_STRUCTURE"
if [[ -n "$TITLE" ]]; then
  validate_cxc_text "--title" "$TITLE"
fi
: > "$COMMAND_LOG"

cat > "$START_SCRIPT" <<CXC
close session
open "$ABS_STRUCTURE"
cartoon #1
color #1 lightgray target c
set bgColor white
lighting soft
graphics silhouettes true
camera ortho
view all pad 0.15
remotecontrol rest start port $PORT json true log true
CXC
if [[ -n "$TITLE" ]]; then
  printf '2dlabels text "%s" xpos 0.035 ypos 0.89 size 30 color black bgColor none\n' "$TITLE" >> "$START_SCRIPT"
fi

"$CHIMERAX_BIN_RESOLVED" --script "$START_SCRIPT" >"$CHIMERAX_LOG" 2>&1 &
CHIMERAX_PID=$!

for i in $(seq 1 40); do
  if curl -fsS --max-time 2 -G --data-urlencode 'command=remotecontrol rest port' "http://127.0.0.1:$PORT/run" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
  if ! kill -0 "$CHIMERAX_PID" >/dev/null 2>&1; then
    printf 'FAIL: ChimeraX exited before REST was ready\n' >&2
    cat "$CHIMERAX_LOG" >&2 || true
    exit 1
  fi
  if [[ "$i" -eq 40 ]]; then
    printf 'FAIL: timed out waiting for ChimeraX REST on 127.0.0.1:%s\n' "$PORT" >&2
    cat "$CHIMERAX_LOG" >&2 || true
    exit 1
  fi
done

cat > "$SESSION_MANIFEST" <<YAML
schema_version: chimerax_control_session_v1
control_session_id: "$SESSION_ID"
pause_point: "session-ready"
started_at_utc: "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
chimerax_executable: "$CHIMERAX_BIN_RESOLVED"
chimerax_pid: $CHIMERAX_PID
source_structure_path: "$ABS_STRUCTURE"
source_structure_sha256: "$(sha256_file "$ABS_STRUCTURE")"
opened_model_id: "#1"
command_log_path: "$COMMAND_LOG"
control:
  host: "127.0.0.1"
  port: $PORT
  rest_stopped: false
  gui_left_open: true
port: $PORT
YAML

printf 'PASS: ChimeraX control session is ready\n'
printf 'PAUSE: session-ready. You can now steer the ChimeraX GUI, or ask the agent to send an allowlisted command.\n'
printf 'INFO: session_manifest=%s\n' "$SESSION_MANIFEST"
printf 'INFO: port=%s\n' "$PORT"
