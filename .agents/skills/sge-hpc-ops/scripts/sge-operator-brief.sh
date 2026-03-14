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
  sge-operator-brief.sh [options]

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

parse_status_card_field() {
  local card="$1"
  local label="$2"
  printf '%s\n' "$card" | sed -n "s/^- $label: //p" | head -n 1
}

parse_advisor_key() {
  local line="$1"
  local key="$2"
  printf '%s\n' "$line" | sed -n "s/.*$key=\([^[:space:]]*\).*/\1/p"
}

main() {
  parse_args "$@"

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  local status_args=("$script_dir/sge-status-card.sh" "--warn-over-running" "$warn_over_running")
  local advisor_args=(
    "$script_dir/sge-submit-shape-advisor.sh"
    "--planned-submits" "$planned_submits"
    "--warn-over-running" "$warn_over_running"
  )

  if [[ "$requires_order" -eq 1 ]]; then
    advisor_args+=("--requires-order")
  fi

  if [[ -n "$qstat_file" ]]; then
    status_args+=("--qstat-file" "$qstat_file")
    advisor_args+=("--qstat-file" "$qstat_file")
  fi

  local status_card advisor_output
  status_card="$("${status_args[@]}")"
  advisor_output="$("${advisor_args[@]}")"

  local advisor_line advisor_reason_line advisor_recommendation_line
  advisor_line="$(printf '%s\n' "$advisor_output" | rg '^ADVISOR ' -m 1 || true)"
  advisor_reason_line="$(printf '%s\n' "$advisor_output" | rg '^REASON ' -m 1 || true)"
  advisor_recommendation_line="$(printf '%s\n' "$advisor_output" | rg '^RECOMMENDATION ' -m 1 || true)"

  local health execution_locus running_line queued_jobs eqw_jobs reason recommendation
  health="$(parse_status_card_field "$status_card" "Health")"
  execution_locus="$(parse_status_card_field "$status_card" "Execution Locus")"
  running_line="$(parse_status_card_field "$status_card" "Running Jobs")"
  queued_jobs="$(parse_status_card_field "$status_card" "Queued Jobs")"
  eqw_jobs="$(parse_status_card_field "$status_card" "Eqw Jobs")"
  reason="$(parse_status_card_field "$status_card" "Reason")"
  recommendation="$(parse_status_card_field "$status_card" "Recommendation")"

  local advisor advisor_reason advisor_recommendation
  advisor="$(parse_advisor_key "$advisor_line" "advisor")"
  advisor_reason="${advisor_reason_line#REASON }"
  advisor_recommendation="${advisor_recommendation_line#RECOMMENDATION }"

  [[ -n "$health" ]] || health="unknown"
  [[ -n "$execution_locus" ]] || execution_locus="unknown"
  [[ -n "$running_line" ]] || running_line="unknown"
  [[ -n "$queued_jobs" ]] || queued_jobs="0"
  [[ -n "$eqw_jobs" ]] || eqw_jobs="0"
  [[ -n "$reason" ]] || reason="status reason unavailable"
  [[ -n "$recommendation" ]] || recommendation="status recommendation unavailable"
  [[ -n "$advisor" ]] || advisor="unknown"
  [[ -n "$advisor_reason" ]] || advisor_reason="advisor reason unavailable"
  [[ -n "$advisor_recommendation" ]] || advisor_recommendation="advisor recommendation unavailable"

  local running_jobs threshold
  running_jobs="$(printf '%s\n' "$running_line" | sed -n 's/^\([0-9][0-9]*\).*/\1/p')"
  threshold="$(printf '%s\n' "$running_line" | sed -n 's/.*threshold \([0-9][0-9]*\).*/\1/p')"
  [[ -n "$running_jobs" ]] || running_jobs=0
  [[ -n "$threshold" ]] || threshold="$warn_over_running"

  local submit_gate next_action
  if [[ "$health" == "red" || "$advisor" == "triage_first" ]]; then
    submit_gate="block"
    next_action="Triage Eqw and failed jobs before any new submission."
  elif [[ "$health" == "yellow" || "$advisor" == "confirm_then_submit" ]]; then
    submit_gate="confirm"
    next_action="Ask for explicit user confirmation, then use advisor-compliant submission shape."
  else
    submit_gate="ready"
    next_action="Proceed with verify-before-submit and template QA preflight."
  fi

  if [[ "$json_output" -eq 1 ]]; then
    printf '{"submit_gate":"%s","health":"%s","execution_locus":"%s","running_jobs":%d,"threshold":%d,"queued_jobs":%d,"eqw_jobs":%d,"planned_submits":%d,"requires_order":%s,"advisor":"%s","status_reason":"%s","advisor_reason":"%s","advisor_recommendation":"%s","next_action":"%s","queue_policy":"respect-queue-no-line-skipping"}\n' \
      "$submit_gate" "$health" "$execution_locus" "$running_jobs" "$threshold" "$queued_jobs" "$eqw_jobs" "$planned_submits" "$([[ "$requires_order" -eq 1 ]] && echo true || echo false)" "$advisor" "$reason" "$advisor_reason" "$advisor_recommendation" "$next_action"
  else
    printf 'HPC Operator Brief\n'
    printf -- '- Submit Gate: %s\n' "$submit_gate"
    printf -- '- Health: %s\n' "$health"
    printf -- '- Execution Locus: %s\n' "$execution_locus"
    printf -- '- Running Jobs: %d (threshold %d)\n' "$running_jobs" "$threshold"
    printf -- '- Queued Jobs: %s\n' "$queued_jobs"
    printf -- '- Eqw Jobs: %s\n' "$eqw_jobs"
    printf -- '- Advisor: %s\n' "$advisor"
    printf -- '- Reason: %s\n' "$advisor_reason"
    printf -- '- Recommendation: %s\n' "$advisor_recommendation"
    printf -- '- Next Action: %s\n' "$next_action"
    printf -- '- Queue Policy: respect queue, do not skip the line\n'
  fi
}

main "$@"
