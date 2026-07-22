#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-55434}"

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

tmp_dir="$(mktemp -d -t chimerax-structure-review-smoke)"
script_path="$tmp_dir/remote_smoke.cxc"
log_path="$tmp_dir/chimerax.log"
printf 'remotecontrol rest start port %s json true log true\n' "$PORT" > "$script_path"

"$CHIMERAX_BIN_RESOLVED" --script "$script_path" >"$log_path" 2>&1 &
chimerax_pid=$!
cleanup() {
  if kill -0 "$chimerax_pid" >/dev/null 2>&1; then
    curl -fsS --max-time 2 -G --data-urlencode 'command=remotecontrol rest stop' "http://127.0.0.1:$PORT/run" >/dev/null 2>&1 || true
    sleep 1
    kill "$chimerax_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

probe_response=""
for _ in $(seq 1 30); do
  if probe_response="$(curl -fsS --max-time 2 -G --data-urlencode 'command=remotecontrol rest port' "http://127.0.0.1:$PORT/run" 2>/dev/null)"; then
    break
  fi
  sleep 0.5
done

if [[ -z "$probe_response" ]]; then
  printf 'FAIL: REST endpoint did not respond on 127.0.0.1:%s\n' "$PORT" >&2
  printf 'INFO: ChimeraX log: %s\n' "$log_path" >&2
  exit 1
fi

python3 - "$probe_response" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
if payload.get("error") is not None:
    raise SystemExit(f"REST probe returned error: {payload['error']}")
PY

curl -fsS --max-time 2 -G --data-urlencode 'command=set bgColor white' "http://127.0.0.1:$PORT/run" >/dev/null
stop_response="$(curl -fsS --max-time 2 -G --data-urlencode 'command=remotecontrol rest stop' "http://127.0.0.1:$PORT/run")"
python3 - "$stop_response" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
if payload.get("error") is not None:
    raise SystemExit(f"REST stop returned error: {payload['error']}")
PY
trap - EXIT
kill "$chimerax_pid" >/dev/null 2>&1 || true
wait "$chimerax_pid" || true

printf 'PASS: ChimeraX REST smoke succeeded on 127.0.0.1:%s\n' "$PORT"
printf 'PASS: REST endpoint stopped\n'
