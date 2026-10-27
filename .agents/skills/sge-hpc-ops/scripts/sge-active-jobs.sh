#!/usr/bin/env bash
set -euo pipefail

max_jobs=12
qstat_file=""
json_output=0

usage() {
  cat <<'USAGE'
Usage:
  sge-active-jobs.sh [options]

Options:
  --max-jobs <int>      Maximum number of jobs to render (default: 12)
  --qstat-file <path>   Read qstat-like output from file (fixture mode)
  --json                Emit JSON output
  -h, --help            Show this help
USAGE
}

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --max-jobs)
        [[ $# -ge 2 ]] || {
          echo "missing value for --max-jobs" >&2
          exit 2
        }
        max_jobs="$2"
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

  if ! [[ "$max_jobs" =~ ^[0-9]+$ ]]; then
    echo "--max-jobs must be a non-negative integer" >&2
    exit 2
  fi

  if [[ -n "$qstat_file" && ! -r "$qstat_file" ]]; then
    echo "qstat fixture file is not readable: $qstat_file" >&2
    exit 2
  fi
}

read_qstat_data() {
  if [[ -n "$qstat_file" ]]; then
    cat "$qstat_file"
    return
  fi

  if command -v qstat >/dev/null 2>&1; then
    qstat -u "${USER:-$(whoami)}" 2>/dev/null || true
    return
  fi

  return 0
}

extract_jobs_tsv() {
  local data="$1"
  if [[ -z "${data//[[:space:]]/}" ]]; then
    return 0
  fi

  awk '
    $1 ~ /^[0-9]+$/ {
      job_id=$1
      name=$3
      state=$5
      queue="-"
      slots="0"
      task_id="-"

      if ($8 ~ /^[0-9]+$/) {
        slots=$8
      } else if ($8 != "") {
        queue=$8
        if ($9 ~ /^[0-9]+$/) {
          slots=$9
        }
        if ($10 != "" && $10 !~ /^[[:space:]]*$/) {
          task_id=$10
        }
      }
      printf "%s\t%s\t%s\t%s\t%s\t%s\n", job_id, name, state, queue, slots, task_id
    }
  ' <<<"$data"
}

main() {
  parse_args "$@"

  local qstat_source qstat_data
  qstat_source="none"
  if [[ -n "$qstat_file" ]]; then
    qstat_source="fixture"
  elif command -v qstat >/dev/null 2>&1; then
    qstat_source="live"
  fi
  qstat_data="$(read_qstat_data)"

  local jobs_tsv
  jobs_tsv="$(extract_jobs_tsv "$qstat_data")"
  local total_jobs shown_jobs
  total_jobs="$(printf '%s\n' "$jobs_tsv" | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
  shown_jobs="$total_jobs"
  if ((shown_jobs > max_jobs)); then
    shown_jobs="$max_jobs"
  fi

  local limited_tsv
  limited_tsv="$(printf '%s\n' "$jobs_tsv" | sed '/^[[:space:]]*$/d' | head -n "$max_jobs" || true)"

  if [[ "$json_output" -eq 1 ]]; then
    printf '{"qstat_source":"%s","total_jobs":%d,"shown_jobs":%d,"jobs":[' "$qstat_source" "$total_jobs" "$shown_jobs"
    local first=1
    while IFS=$'\t' read -r job_id name state queue slots task_id; do
      [[ -n "$job_id" ]] || continue
      if [[ "$first" -eq 0 ]]; then
        printf ','
      fi
      first=0
      printf '{"job_id":"%s","name":"%s","state":"%s","queue":"%s","slots":%d,"task_id":"%s"}' \
        "$(json_escape "$job_id")" \
        "$(json_escape "$name")" \
        "$(json_escape "$state")" \
        "$(json_escape "$queue")" \
        "$slots" \
        "$(json_escape "$task_id")"
    done <<<"$limited_tsv"
    printf ']}\n'
  else
    printf 'ACTIVE_JOBS total_jobs=%d shown_jobs=%d qstat_source=%s\n' "$total_jobs" "$shown_jobs" "$qstat_source"
    printf 'job_id\tname\tstate\tqueue\tslots\ttask_id\n'
    printf '%s\n' "$limited_tsv"
  fi
}

main "$@"
