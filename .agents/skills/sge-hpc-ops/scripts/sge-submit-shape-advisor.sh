#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./rg_compat.sh
source "$SCRIPT_DIR/rg_compat.sh"

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
  jobs_line="$(printf '%s\n' "$status_output" | rg -m 1 '^JOBS ' || true)"

  running="$(extract_field "$jobs_line" "running_jobs")"
  eqw="$(extract_field "$jobs_line" "eqw_jobs")"
  threshold="$(extract_field "$jobs_line" "threshold")"

  [[ -n "$running" ]] || running=0
  [[ -n "$eqw" ]] || eqw=0
  [[ -n "$threshold" ]] || threshold="$warn_over_running"

  local advisor reason recommended_action
  if ((eqw > 0)); then
    advisor="hold"
    reason="eqw_present"
    recommended_action="triage_eqw_before_submit"
  else
    if ((planned_submits <= 1)); then
      advisor="single"
      reason="single_submit"
      recommended_action="submit_single"
    elif ((requires_order == 1)); then
      advisor="hold_jid"
      reason="ordered_pipeline"
      recommended_action="dependency_chain"
    else
      advisor="array"
      reason="multi_submit"
      recommended_action="array_or_dependency_chain"
    fi
  fi

  if ((running > threshold)) && ((eqw == 0)); then
    reason="${reason};running_jobs_over_threshold"
    case "$advisor" in
      single)
        recommended_action="confirm_then_submit_single"
        ;;
      hold_jid)
        recommended_action="confirm_then_dependency_chain"
        ;;
      array)
        recommended_action="confirm_then_array_or_dependency_chain"
        ;;
    esac
  fi

  if [[ "$json_output" -eq 1 ]]; then
    printf '{"advisor":"%s","reason":"%s","recommended_action":"%s","running_jobs":%d,"threshold":%d,"planned_submits":%d,"requires_order":%s,"queue_policy":"respect_queue"}\n' \
      "$advisor" "$reason" "$recommended_action" "$running" "$threshold" "$planned_submits" "$([[ "$requires_order" -eq 1 ]] && echo true || echo false)"
  else
    printf 'ADVISOR advisor=%s running_jobs=%d threshold=%d planned_submits=%d requires_order=%s queue_policy=respect_queue\n' \
      "$advisor" "$running" "$threshold" "$planned_submits" "$([[ "$requires_order" -eq 1 ]] && echo yes || echo no)"
    case "$reason" in
      eqw_present)
        printf 'REASON Eqw jobs present\n'
        ;;
      single_submit)
        printf 'REASON single submit under low queue pressure\n'
        ;;
      'single_submit;running_jobs_over_threshold')
        printf 'REASON single submit requested while running_jobs exceeds threshold\n'
        ;;
      ordered_pipeline)
        printf 'REASON ordered multi-submit workload\n'
        ;;
      'ordered_pipeline;running_jobs_over_threshold')
        printf 'REASON ordered multi-submit workload under high running-job pressure\n'
        ;;
      multi_submit)
        printf 'REASON independent multi-submit workload\n'
        ;;
      'multi_submit;running_jobs_over_threshold')
        printf 'REASON independent multi-submit workload under high running-job pressure\n'
        ;;
      *)
        printf 'REASON %s\n' "$reason"
        ;;
    esac
    case "$recommended_action" in
      triage_eqw_before_submit)
        printf 'RECOMMENDATION Resolve Eqw before additional submissions.\n'
        ;;
      submit_single)
        printf 'RECOMMENDATION Proceed with single submit after verify gate.\n'
        ;;
      confirm_then_submit_single)
        printf 'RECOMMENDATION Confirm one additional submit and respect queue fairness.\n'
        ;;
      dependency_chain)
        printf 'RECOMMENDATION Use dependency chain with -hold_jid.\n'
        ;;
      confirm_then_dependency_chain)
        printf 'RECOMMENDATION Confirm before using dependency chain with -hold_jid.\n'
        ;;
      array_or_dependency_chain)
        printf 'RECOMMENDATION Use array job for scheduler efficiency.\n'
        ;;
      confirm_then_array_or_dependency_chain)
        printf 'RECOMMENDATION Confirm before converting to array job and avoid burst submits.\n'
        ;;
      *)
        printf 'RECOMMENDATION %s\n' "$recommended_action"
        ;;
    esac
  fi
}

main "$@"
