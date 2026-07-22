#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEMO_STRUCTURE="$SKILL_DIR/assets/demo_structure.pdb"
PORT=""
OUTPUT_DIR=""
KEEP_REST=0
CLOSE_AFTER=0

usage() {
  cat <<'USAGE'
Usage: chimerax-live-demo.sh [--port PORT] [--output-dir DIR] [--keep-rest] [--close-after]

Open the packaged demo structure in a visible ChimeraX session, then use one
localhost REST session to change the view, show side-chain atoms, add a surface,
and capture PNG/session/provenance artifacts.

Default: stop REST after capture and leave the ChimeraX window visible.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --keep-rest)
      KEEP_REST=1
      shift
      ;;
    --close-after)
      CLOSE_AFTER=1
      shift
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

if [[ ! -f "$DEMO_STRUCTURE" ]]; then
  printf 'FAIL: missing demo structure: %s\n' "$DEMO_STRUCTURE" >&2
  exit 1
fi

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

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$(mktemp -d -t chimerax-structure-review-live)"
fi
mkdir -p "$OUTPUT_DIR"
COMMAND_LOG="$OUTPUT_DIR/live_session.commands.jsonl"
LIVE_MANIFEST="$OUTPUT_DIR/live_session_manifest.yaml"
START_SCRIPT="$OUTPUT_DIR/start_live_session.cxc"
CHIMERAX_LOG="$OUTPUT_DIR/chimerax.log"
CAPTURE_DIR="$OUTPUT_DIR/capture"
mkdir -p "$CAPTURE_DIR"

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

json_log() {
  local key="$1"
  local command="$2"
  local response="$3"
  python3 - "$COMMAND_LOG" "$key" "$command" "$response" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
command = sys.argv[3]
raw_response = sys.argv[4]
try:
    response = json.loads(raw_response)
except json.JSONDecodeError:
    response = {"raw": raw_response}
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"key": key, "command": command, "response": response}, sort_keys=True) + "\n")
PY
}

send_live_command() {
  local key="$1"
  local command="$2"
  local response
  response="$("$SCRIPT_DIR/chimerax-send-command.py" --port "$PORT" --command "$command" --timeout-seconds 20)"
  json_log "$key" "$command" "$response"
}

cat > "$START_SCRIPT" <<CXC
close session
open "$DEMO_STRUCTURE"
cartoon #1
color #1 lightgray target c
set bgColor white
lighting soft
graphics silhouettes true
camera ortho
view
remotecontrol rest start port $PORT json true log true
CXC

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

send_live_command set_background 'set bgColor white'
send_live_command lighting 'lighting soft'
send_live_command silhouettes 'graphics silhouettes true'
send_live_command cartoon_style 'cartoon style width 1.2 thick 0.28'
send_live_command show_sidechain_atoms 'show #1/A:2-4 & sidechain atoms'
send_live_command style_sidechain_atoms 'style #1/A:2-4 & sidechain stick'
send_live_command color_sidechain_atoms 'color #1/A:2-4 & sidechain cornflower blue target ab'
send_live_command surface_selection 'surface #1 color lightgray transparency 65'
send_live_command turn_view_y 'turn y 30 16'
send_live_command wait_turn_y 'wait 16'
send_live_command turn_view_x 'turn x -15 10'
send_live_command wait_turn_x 'wait 10'
send_live_command fit_view 'view all pad 0.18'
send_live_command title_label '2dlabels text "ChimeraX live skill dogfood" xpos 0.035 ypos 0.89 size 30 color black bgColor none'

capture_args=(
  --port "$PORT"
  --output-dir "$CAPTURE_DIR"
  --pose-id live_demo_pose
  --title "ChimeraX live skill dogfood"
  --width 1200
  --height 900
  --supersample 1
  --chimerax-executable "$CHIMERAX_BIN_RESOLVED"
  --structure-path "$DEMO_STRUCTURE"
  --opened-model-id "#1"
)
if [[ "$KEEP_REST" -eq 1 ]]; then
  capture_args+=(--keep-rest-open)
fi
"$SCRIPT_DIR/chimerax-capture-pose.py" "${capture_args[@]}"

REST_STOPPED=false
if [[ "$KEEP_REST" -eq 0 ]]; then
  REST_STOPPED=true
fi
GUI_LEFT_OPEN=true
if [[ "$CLOSE_AFTER" -eq 1 ]]; then
  kill "$CHIMERAX_PID" >/dev/null 2>&1 || true
  GUI_LEFT_OPEN=false
fi

cat > "$LIVE_MANIFEST" <<YAML
schema_version: chimerax_live_session_manifest_v1
started_at_utc: "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
chimerax_executable: "$CHIMERAX_BIN_RESOLVED"
chimerax_pid: $CHIMERAX_PID
source_structure_path: "$DEMO_STRUCTURE"
source_structure_sha256: "$(sha256_file "$DEMO_STRUCTURE")"
control:
  host: "127.0.0.1"
  port: $PORT
  rest_stopped: $REST_STOPPED
  gui_left_open: $GUI_LEFT_OPEN
outputs:
  output_dir: "$OUTPUT_DIR"
  command_log_path: "$COMMAND_LOG"
  command_log_sha256: "$(sha256_file "$COMMAND_LOG")"
  capture_manifest_path: "$CAPTURE_DIR/live_demo_pose.pose_manifest.yaml"
  capture_manifest_sha256: "$(sha256_file "$CAPTURE_DIR/live_demo_pose.pose_manifest.yaml")"
  capture_image_path: "$CAPTURE_DIR/live_demo_pose.png"
  capture_session_path: "$CAPTURE_DIR/live_demo_pose.cxs"
YAML

printf 'PASS: live ChimeraX demo completed on 127.0.0.1:%s\n' "$PORT"
printf 'INFO: output_dir=%s\n' "$OUTPUT_DIR"
printf 'INFO: live_manifest=%s\n' "$LIVE_MANIFEST"
if [[ "$GUI_LEFT_OPEN" == true ]]; then
  printf 'INFO: ChimeraX GUI remains open for inspection; REST stopped=%s\n' "$REST_STOPPED"
fi
