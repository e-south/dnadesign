#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./rg_compat.sh
source "$SCRIPT_DIR/rg_compat.sh"

warn_over_running=3
qstat_file=""
json_output=0

usage() {
  cat <<'USAGE'
Usage:
  sge-status-card.sh [options]

Options:
  --warn-over-running <int>   Warning threshold for running jobs (default: 3)
  --qstat-file <path>         Read qstat-like output from file (fixture mode)
  --json                      Emit JSON output
  -h, --help                  Show this help
USAGE
}

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

json_int_or_null() {
  local value="${1:-}"
  if [[ "$value" =~ ^[0-9]+$ ]]; then
    printf '%s' "$value"
  else
    printf 'null'
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --warn-over-running)
        [[ $# -ge 2 ]] || {
          echo "missing value for --warn-over-running" >&2
          exit 2
        }
        warn_over_running="$2"
        shift 2
        ;;
      --qstat-file)
        [[ $# -ge 2 ]] || {
          echo "missing value for --qstat-file" >&2
          exit 2
        }
        qstat_file="$2"
        shift 2
        ;;
      --json)
        json_output=1
        shift
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *)
        echo "unknown argument: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
  done

  if ! [[ "$warn_over_running" =~ ^[0-9]+$ ]]; then
    echo "--warn-over-running must be a non-negative integer" >&2
    exit 2
  fi

  if [[ -n "$qstat_file" && ! -r "$qstat_file" ]]; then
    echo "qstat fixture file is not readable: $qstat_file" >&2
    exit 2
  fi
}

extract_field() {
  local line="$1"
  local key="$2"
  printf '%s\n' "$line" | sed -n "s/.*$key=\([^[:space:]]*\).*/\1/p"
}

main() {
  parse_args "$@"

  local script_dir status_cmd status_output
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  status_cmd=("$script_dir/sge-session-status.sh" "--warn-over-running" "$warn_over_running")
  if [[ -n "$qstat_file" ]]; then
    status_cmd+=("--qstat-file" "$qstat_file")
  fi

  status_output="$("${status_cmd[@]}")"

  local session_line tools_line jobs_line locus queue_probe running queued eqw threshold
  session_line="$(printf '%s\n' "$status_output" | rg -m 1 '^SESSION ' || true)"
  tools_line="$(printf '%s\n' "$status_output" | rg -m 1 '^SGE_TOOLS ' || true)"
  jobs_line="$(printf '%s\n' "$status_output" | rg -m 1 '^JOBS ' || true)"

  locus="$(extract_field "$session_line" "execution_locus_guess")"
  queue_probe="$(extract_field "$tools_line" "queue_probe")"
  running="$(extract_field "$jobs_line" "running_jobs")"
  queued="$(extract_field "$jobs_line" "queued_jobs")"
  eqw="$(extract_field "$jobs_line" "eqw_jobs")"
  threshold="$(extract_field "$jobs_line" "threshold")"

  [[ -n "$locus" ]] || locus="unknown"
  [[ -n "$queue_probe" ]] || queue_probe="unknown"
  [[ -n "$running" ]] || running="unknown"
  [[ -n "$queued" ]] || queued="unknown"
  [[ -n "$eqw" ]] || eqw="unknown"
  [[ -n "$threshold" ]] || threshold="$warn_over_running"

  local health reason recommendation
  if [[ "$queue_probe" == "host_denied" ]]; then
    health="red"
    reason="current host is not submit-capable"
    recommendation="Use a submit-capable SCC shell or OnDemand app shell."
  elif [[ "$queue_probe" != "ok" ]]; then
    health="yellow"
    reason="queue probe unavailable"
    recommendation="Re-probe on a submit-capable shell before treating queue counts as authoritative."
  elif [[ "$eqw" =~ ^[0-9]+$ ]] && ((eqw > 0)); then
    health="red"
    reason="Eqw jobs detected"
    recommendation="Triage Eqw jobs before additional submissions."
  elif [[ "$running" =~ ^[0-9]+$ ]] && ((running > threshold)); then
    health="yellow"
    reason="running_jobs exceeds threshold"
    recommendation="Confirm before additional submissions and prefer arrays or -hold_jid."
  else
    health="green"
    reason="within threshold and no Eqw jobs"
    recommendation="Proceed with verify-before-submit gate."
  fi

  if [[ "$json_output" -eq 1 ]]; then
    local queue_policy
    queue_policy="respect_queue"
    if [[ "$queue_probe" == "host_denied" ]]; then
      queue_policy="submit_host_required"
    fi
    printf '{"health":"%s","execution_locus":"%s","running_jobs":%s,"queued_jobs":%s,"eqw_jobs":%s,"threshold":%d,"reason":"%s","recommendation":"%s","queue_probe":"%s","queue_policy":"%s"}\n' \
      "$health" "$locus" "$(json_int_or_null "$running")" "$(json_int_or_null "$queued")" "$(json_int_or_null "$eqw")" "$threshold" "$(json_escape "$reason")" "$(json_escape "$recommendation")" "$queue_probe" "$queue_policy"
  else
    local queue_policy
    queue_policy="respect_queue"
    if [[ "$queue_probe" == "host_denied" ]]; then
      queue_policy="submit host required"
    fi
    printf 'HPC Status Card\n'
    printf -- '- Health: %s\n' "$health"
    printf -- '- Execution Locus: %s\n' "$locus"
    printf -- '- Queue Probe: %s\n' "$queue_probe"
    printf -- '- Running Jobs: %s (threshold %d)\n' "$running" "$threshold"
    printf -- '- Queued Jobs: %s\n' "$queued"
    printf -- '- Eqw Jobs: %s\n' "$eqw"
    printf -- '- Reason: %s\n' "$reason"
    printf -- '- Recommendation: %s\n' "$recommendation"
    printf -- '- Queue Policy: %s\n' "$queue_policy"
  fi
}

main "$@"
