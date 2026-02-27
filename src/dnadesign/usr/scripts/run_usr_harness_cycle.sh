#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# dnadesign
# dnadesign/src/dnadesign/usr/scripts/run_usr_harness_cycle.sh
#
# Runs the deterministic USR preflight -> run -> verify harness cycle.
#
# Module Author(s): Eric J. South
# ------------------------------------------------------------------------------

set -euo pipefail

REPORT_PATH="${USR_HARNESS_REPORT_PATH:-}"
STARTED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
STEP_LOG="$(mktemp -t usr-harness-steps.XXXXXX)"
FAILED_STEP=""

write_report() {
  local exit_code="$1"
  if [[ -n "${REPORT_PATH}" ]]; then
    mkdir -p "$(dirname "${REPORT_PATH}")"
    python3 - "${REPORT_PATH}" "${exit_code}" "${STARTED_AT}" "${STEP_LOG}" "${FAILED_STEP}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

report_path = pathlib.Path(sys.argv[1])
exit_code = int(sys.argv[2])
started_at = sys.argv[3]
step_log = pathlib.Path(sys.argv[4])
failed_step = sys.argv[5] or None

steps = []
for raw in step_log.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    name, status, duration, command = raw.split("\t", 3)
    status_code = int(status)
    steps.append(
        {
            "name": name,
            "status": "success" if status_code == 0 else "failure",
            "exit_code": status_code,
            "duration_seconds": int(duration),
            "command": command,
        }
    )

payload = {
    "started_at": started_at,
    "finished_at": datetime.now(tz=timezone.utc).isoformat(),
    "status": "success" if exit_code == 0 else "failure",
    "exit_code": exit_code,
    "failed_step": failed_step,
    "steps": steps,
}
report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  fi
  rm -f "${STEP_LOG}"
}

trap 'write_report "$?"' EXIT

run_step() {
  local name="$1"
  shift
  local started_epoch
  local finished_epoch
  local duration
  local status
  local command_text="$*"
  started_epoch="$(date +%s)"
  echo "[${name}] ${command_text}"
  if "$@"; then
    status=0
  else
    status=$?
    if [[ -z "${FAILED_STEP}" ]]; then
      FAILED_STEP="${name}"
    fi
  fi
  finished_epoch="$(date +%s)"
  duration="$((finished_epoch - started_epoch))"
  printf "%s\t%s\t%s\t%s\n" "${name}" "${status}" "${duration}" "${command_text}" >> "${STEP_LOG}"
  return "${status}"
}

run_step "preflight-cli-help" bash -lc "uv run usr --help >/dev/null"
run_step \
  "preflight-sync-focused-tests" \
  uv run pytest -q \
  src/dnadesign/usr/tests/test_sync_iterative_batch_flow.py \
  src/dnadesign/usr/tests/test_sync_schema_adversarial.py \
  src/dnadesign/usr/tests/test_sync_target_modes.py
run_step "run-full-usr-tests" uv run pytest -q src/dnadesign/usr/tests
run_step "verify-ruff-check" uv run ruff check src/dnadesign/usr/src src/dnadesign/usr/tests
run_step "verify-ruff-format" uv run ruff format --check src/dnadesign/usr/src src/dnadesign/usr/tests
run_step "verify-docs-checks" uv run python -m dnadesign.devtools.docs_checks
