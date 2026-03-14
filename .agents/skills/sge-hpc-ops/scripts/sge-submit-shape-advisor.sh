#!/usr/bin/env bash
set -euo pipefail

warn_over_running=3
planned_submits=1
requires_order=0
qstat_file=""
json_output=0

usage() {
  cat <<'USAGE'
Usage:
  sge-submit-shape-advisor.sh [options]

Options:
  --planned-submits <int>      Number of submits being planned (default: 1)
  --requires-order             Jobs must execute in strict order
  --warn-over-running <int>    Running-jobs warning threshold (default: 3)
  --qstat-file <path>          Read qstat-like output from file (fixture mode)
  --json                       Emit JSON output
  -h, --help                   Show this help
USAGE
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --planned-submits)
        [[ $# -ge 2 ]] || {
          echo "missing value for --planned-submits" >&2
          exit 2
        }
        planned_submits="$2"
        shift 2
        ;;
      --requires-order)
        requires_order=1
        shift
        ;;
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

  if ! [[ "$planned_submits" =~ ^[0-9]+$ ]]; then
    echo "--planned-submits must be a non-negative integer" >&2
    exit 2
  fi

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

  local jobs_line running eqw threshold
  jobs_line="$(printf '%s\n' "$status_output" | rg '^JOBS ' -m 1 || true)"

  running="$(extract_field "$jobs_line" "running_jobs")"
  eqw="$(extract_field "$jobs_line" "eqw_jobs")"
  threshold="$(extract_field "$jobs_line" "threshold")"

  [[ -n "$running" ]] || running=0
  [[ -n "$eqw" ]] || eqw=0
  [[ -n "$threshold" ]] || threshold="$warn_over_running"

  local advisor reason recommended_action
  if ((eqw > 0)); then
    advisor="triage_first"
    reason="Eqw jobs present"
    recommended_action="Resolve Eqw before additional submissions."
  elif ((running > threshold)); then
    if ((planned_submits <= 1)); then
      advisor="confirm_then_submit"
      reason="high running load for single additional submit"
      recommended_action="Confirm one additional submit and respect queue fairness."
    elif ((requires_order == 1)); then
      advisor="hold_jid"
      reason="ordered multi-submit under high load"
      recommended_action="Use dependency chain with -hold_jid and avoid burst submits."
    else
      advisor="array"
      reason="independent multi-submit under high load"
      recommended_action="Convert to array job and avoid many independent submits."
    fi
  else
    if ((planned_submits <= 1)); then
      advisor="single_submit"
      reason="low pressure single submit"
      recommended_action="Proceed with single submit after verify gate."
    elif ((requires_order == 1)); then
      advisor="hold_jid"
      reason="ordered multi-submit"
      recommended_action="Use dependency chain with -hold_jid."
    else
      advisor="array"
      reason="independent multi-submit"
      recommended_action="Use array job for scheduler efficiency."
    fi
  fi

  if [[ "$json_output" -eq 1 ]]; then
    printf '{"advisor":"%s","reason":"%s","recommended_action":"%s","running_jobs":%d,"threshold":%d,"planned_submits":%d,"requires_order":%s,"queue_policy":"respect-queue-no-line-skipping"}\n' \
      "$advisor" "$reason" "$recommended_action" "$running" "$threshold" "$planned_submits" "$([[ "$requires_order" -eq 1 ]] && echo true || echo false)"
  else
    printf 'ADVISOR advisor=%s running_jobs=%d threshold=%d planned_submits=%d requires_order=%s queue_policy=respect-queue-no-line-skipping\n' \
      "$advisor" "$running" "$threshold" "$planned_submits" "$([[ "$requires_order" -eq 1 ]] && echo yes || echo no)"
    printf 'REASON %s\n' "$reason"
    printf 'RECOMMENDATION %s\n' "$recommended_action"
  fi
}

main "$@"
