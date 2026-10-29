#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_MANIFEST=""
CLOSE_GUI=0

usage() {
  cat <<'USAGE'
Usage: chimerax-session-stop.sh --session-manifest PATH [--close-gui]

Stop REST control for a ChimeraX control session. Optionally close the GUI
process after stopping REST.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-manifest)
      SESSION_MANIFEST="$2"
      shift 2
      ;;
    --close-gui)
      CLOSE_GUI=1
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

if [[ -z "$SESSION_MANIFEST" || ! -f "$SESSION_MANIFEST" ]]; then
  printf 'FAIL: --session-manifest must name an existing file\n' >&2
  exit 1
fi

PID="$(sed -n 's/^chimerax_pid: //p' "$SESSION_MANIFEST" | head -n 1)"
PORT="$(sed -n 's/^port: //p' "$SESSION_MANIFEST" | head -n 1)"
CHIMERAX_EXECUTABLE="$(sed -n 's/^chimerax_executable: //p' "$SESSION_MANIFEST" | head -n 1)"
CHIMERAX_EXECUTABLE="${CHIMERAX_EXECUTABLE#\"}"
CHIMERAX_EXECUTABLE="${CHIMERAX_EXECUTABLE%\"}"
if [[ -z "$PID" || -z "$PORT" || -z "$CHIMERAX_EXECUTABLE" ]]; then
  printf 'FAIL: manifest missing chimerax_executable, chimerax_pid, or port: %s\n' "$SESSION_MANIFEST" >&2
  exit 1
fi

validate_close_target() {
  if ! kill -0 "$PID" >/dev/null 2>&1; then
    printf 'FAIL: manifest ChimeraX process is not running: pid=%s\n' "$PID" >&2
    exit 1
  fi

  local process_command expected_name expected_name_lower
  expected_name="$(basename "$CHIMERAX_EXECUTABLE")"
  expected_name_lower="$(printf '%s' "$expected_name" | tr '[:upper:]' '[:lower:]')"
  if [[ "$CHIMERAX_EXECUTABLE" != /* || "$expected_name_lower" != "chimerax" ]]; then
    printf 'FAIL: manifest chimerax_executable must be an absolute ChimeraX executable path: %s\n' "$CHIMERAX_EXECUTABLE" >&2
    exit 1
  fi
  process_command="$(ps -ww -p "$PID" -o command= 2>/dev/null || true)"
  case "$process_command" in
    "$CHIMERAX_EXECUTABLE"|"$CHIMERAX_EXECUTABLE "*) ;;
    *)
      printf 'FAIL: manifest pid does not identify the declared ChimeraX executable: pid=%s\n' "$PID" >&2
      exit 1
      ;;
  esac

  if ! command -v lsof >/dev/null 2>&1; then
    printf 'FAIL: lsof is required to verify the ChimeraX REST owner before closing the GUI\n' >&2
    exit 1
  fi
  if ! lsof -nP -a -p "$PID" -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | grep -Fxq "$PID"; then
    printf 'FAIL: manifest pid does not own the recorded ChimeraX REST port: pid=%s port=%s\n' "$PID" "$PORT" >&2
    exit 1
  fi
}

if ! [[ "$PID" =~ ^[1-9][0-9]*$ ]] || (( PID <= 1 )); then
  printf 'FAIL: manifest chimerax_pid must be a positive process id greater than 1: %s\n' "$PID" >&2
  exit 1
fi
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1024 || PORT > 65535 )); then
  printf 'FAIL: manifest port must be an integer from 1024 to 65535: %s\n' "$PORT" >&2
  exit 1
fi

if [[ "$CLOSE_GUI" -eq 1 ]]; then
  validate_close_target
fi

if ! "$SCRIPT_DIR/chimerax-send-command.py" \
  --session-manifest "$SESSION_MANIFEST" \
  --command 'remotecontrol rest stop' >/dev/null 2>&1; then
  printf 'FAIL: ChimeraX REST stop command was not accepted at 127.0.0.1:%s\n' "$PORT" >&2
  exit 1
fi
printf 'PASS: REST stop command sent for 127.0.0.1:%s\n' "$PORT"

if [[ "$CLOSE_GUI" -eq 1 ]]; then
  if ! kill "$PID" >/dev/null 2>&1; then
    printf 'FAIL: could not request GUI close for pid=%s\n' "$PID" >&2
    exit 1
  fi
  printf 'PASS: requested GUI close for pid=%s\n' "$PID"
else
  printf 'PAUSE: stop-or-continue. REST is stopped; GUI remains open if the process is still running.\n'
fi
