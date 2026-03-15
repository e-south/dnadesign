#!/usr/bin/env bash
set -euo pipefail

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

  local session_line jobs_line locus running queued eqw threshold
  session_line="$(printf '%s\n' "$status_output" | rg '^SESSION ' -m 1 || true)"
  jobs_line="$(printf '%s\n' "$status_output" | rg '^JOBS ' -m 1 || true)"

  locus="$(extract_field "$session_line" "execution_locus_guess")"
  running="$(extract_field "$jobs_line" "running_jobs")"
  queued="$(extract_field "$jobs_line" "queued_jobs")"
  eqw="$(extract_field "$jobs_line" "eqw_jobs")"
  threshold="$(extract_field "$jobs_line" "threshold")"

  [[ -n "$locus" ]] || locus="unknown"
  [[ -n "$running" ]] || running=0
  [[ -n "$queued" ]] || queued=0
  [[ -n "$eqw" ]] || eqw=0
  [[ -n "$threshold" ]] || threshold="$warn_over_running"

  local health reason recommendation
  if ((eqw > 0)); then
    health="red"
    reason="Eqw jobs detected"
    recommendation="Triage Eqw jobs before additional submissions."
  elif ((running > threshold)); then
    health="yellow"
    reason="running_jobs exceeds threshold"
    recommendation="Confirm before additional submissions and prefer arrays or -hold_jid."
  else
    health="green"
    reason="within threshold and no Eqw jobs"
    recommendation="Proceed with verify-before-submit gate."
  fi

  if [[ "$json_output" -eq 1 ]]; then
    printf '{"health":"%s","execution_locus":"%s","running_jobs":%d,"queued_jobs":%d,"eqw_jobs":%d,"threshold":%d,"reason":"%s","recommendation":"%s","queue_policy":"respect-queue-no-line-skipping"}\n' \
      "$health" "$locus" "$running" "$queued" "$eqw" "$threshold" "$reason" "$recommendation"
  else
    printf 'HPC Status Card\n'
    printf -- '- Health: %s\n' "$health"
    printf -- '- Execution Locus: %s\n' "$locus"
    printf -- '- Running Jobs: %d (threshold %d)\n' "$running" "$threshold"
    printf -- '- Queued Jobs: %d\n' "$queued"
    printf -- '- Eqw Jobs: %d\n' "$eqw"
    printf -- '- Reason: %s\n' "$reason"
    printf -- '- Recommendation: %s\n' "$recommendation"
    printf -- '- Queue Policy: respect queue, do not skip the line\n'
  fi
}

main "$@"
