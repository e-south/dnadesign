#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_MANIFEST=""

usage() {
  cat <<'USAGE'
Usage: chimerax-session-status.sh --session-manifest PATH

Check whether a control-session manifest still points to a live ChimeraX process
and REST endpoint.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-manifest)
      SESSION_MANIFEST="$2"
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

if kill -0 "$PID" >/dev/null 2>&1; then
  printf 'PASS: ChimeraX process is running: pid=%s\n' "$PID"
else
  printf 'FAIL: ChimeraX process is not running: pid=%s\n' "$PID" >&2
  exit 1
fi

if "$SCRIPT_DIR/chimerax-send-command.py" --session-manifest "$SESSION_MANIFEST" --command 'remotecontrol rest port' >/dev/null 2>&1; then
  printf 'PASS: REST endpoint responds: 127.0.0.1:%s\n' "$PORT"
  printf 'PAUSE: user-steering or agent-action. The shared session is ready for the next visible operation.\n'
else
  printf 'FAIL: REST endpoint does not respond: 127.0.0.1:%s\n' "$PORT" >&2
  exit 1
fi
