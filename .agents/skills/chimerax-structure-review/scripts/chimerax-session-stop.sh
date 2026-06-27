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
if [[ -z "$PID" || -z "$PORT" ]]; then
  printf 'FAIL: manifest missing chimerax_pid or port: %s\n' "$SESSION_MANIFEST" >&2
  exit 1
fi

"$SCRIPT_DIR/chimerax-send-command.py" --session-manifest "$SESSION_MANIFEST" --command 'remotecontrol rest stop' >/dev/null 2>&1 || true
printf 'PASS: REST stop command sent for 127.0.0.1:%s\n' "$PORT"

if [[ "$CLOSE_GUI" -eq 1 ]]; then
  kill "$PID" >/dev/null 2>&1 || true
  printf 'PASS: requested GUI close for pid=%s\n' "$PID"
else
  printf 'PAUSE: stop-or-continue. REST is stopped; GUI remains open if the process is still running.\n'
fi
