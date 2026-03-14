#!/usr/bin/env bash
set -euo pipefail

max_runtime_hours=12
require_project_flag=0
require_mem_per_core=0
allow_runtime_over=0
declare -a templates

usage() {
  cat <<'USAGE'
Usage:
  qa-sge-submit-preflight.sh --template <path> [--template <path> ...] [options]

Options:
  --template <path>           Path to qsub template (repeatable)
  --max-runtime-hours <int>   Max allowed h_rt in hours before failure (default: 12)
  --allow-runtime-over        Allow templates to exceed max runtime threshold
  --require-project-flag      Require "#$ -P" in template
  --require-mem-per-core      Require mem_per_core when omp slots > 1
  -h, --help                  Show this help
USAGE
}

fail_count=0
warn_count=0

fail_msg() {
  local template="$1"
  local msg="$2"
  printf 'FAIL [%s] %s\n' "$template" "$msg"
  fail_count=$((fail_count + 1))
}

warn_msg() {
  local template="$1"
  local msg="$2"
  printf 'WARN [%s] %s\n' "$template" "$msg"
  warn_count=$((warn_count + 1))
}

pass_msg() {
  local template="$1"
  local msg="$2"
  printf 'PASS [%s] %s\n' "$template" "$msg"
}

to_seconds() {
  local hms="$1"
  local h m s
  IFS=':' read -r h m s <<<"$hms"
  echo $((10#$h * 3600 + 10#$m * 60 + 10#$s))
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --template)
        [[ $# -ge 2 ]] || {
          echo "missing value for --template" >&2
          exit 2
        }
        templates+=("$2")
        shift 2
        ;;
      --max-runtime-hours)
        [[ $# -ge 2 ]] || {
          echo "missing value for --max-runtime-hours" >&2
          exit 2
        }
        max_runtime_hours="$2"
        shift 2
        ;;
      --allow-runtime-over)
        allow_runtime_over=1
        shift
        ;;
      --require-project-flag)
        require_project_flag=1
        shift
        ;;
      --require-mem-per-core)
        require_mem_per_core=1
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

  if [[ ${#templates[@]} -eq 0 ]]; then
    echo "at least one --template is required" >&2
    usage >&2
    exit 2
  fi
}

check_template() {
  local template="$1"
  local is_notify_watcher=0

  if [[ ! -r "$template" ]]; then
    fail_msg "$template" "template is not readable"
    return
  fi

  if rg -q 'notify[[:space:]]+usr-events[[:space:]]+watch' "$template"; then
    is_notify_watcher=1
  fi

  local shebang
  shebang="$(head -n 1 "$template" || true)"
  if [[ ! "$shebang" =~ ^#! ]]; then
    fail_msg "$template" "missing shebang line"
  else
    pass_msg "$template" "shebang present"
  fi

  if rg -q '^[[:space:]]*module[[:space:]]+load[[:space:]]+' "$template"; then
    if [[ "$shebang" != *"-l"* ]]; then
      fail_msg "$template" "module usage requires login shell shebang (#!/bin/bash -l)"
    else
      pass_msg "$template" "module-safe login shell shebang present"
    fi
  fi

  if rg -q '^[[:space:]]*#\$[[:space:]]*-l[[:space:]]+h_rt=' "$template"; then
    pass_msg "$template" "explicit h_rt found"
    local runtime_raw runtime_sec max_runtime_sec
    runtime_raw="$(rg -o --pcre2 '(?<=h_rt=)\d{1,3}:\d{2}:\d{2}' "$template" -m 1 || true)"
    if [[ -n "$runtime_raw" ]]; then
      runtime_sec="$(to_seconds "$runtime_raw")"
      max_runtime_sec=$((max_runtime_hours * 3600))
      if ((runtime_sec > max_runtime_sec)) && [[ "$allow_runtime_over" -ne 1 ]]; then
        if [[ "$is_notify_watcher" -eq 1 ]]; then
          warn_msg "$template" "runtime $runtime_raw exceeds ${max_runtime_hours}h threshold for watcher template; expect runbook submit to override -l h_rt for batch-coupled sessions"
        else
          fail_msg "$template" "runtime $runtime_raw exceeds ${max_runtime_hours}h threshold"
        fi
      elif ((runtime_sec > max_runtime_sec)); then
        warn_msg "$template" "runtime $runtime_raw exceeds ${max_runtime_hours}h threshold but override is enabled"
      else
        pass_msg "$template" "runtime $runtime_raw within threshold"
      fi
    else
      warn_msg "$template" "could not parse h_rt value"
    fi
  else
    fail_msg "$template" "missing explicit h_rt"
  fi

  if [[ "$require_project_flag" -eq 1 ]]; then
    if rg -q '^[[:space:]]*#\$[[:space:]]*-P[[:space:]]+' "$template"; then
      pass_msg "$template" "project flag present"
    else
      fail_msg "$template" "missing required project flag (\"#$ -P\")"
    fi
  fi

  if rg -q '^[[:space:]]*#\$[[:space:]]*-t[[:space:]]+' "$template"; then
    if rg -q 'SGE_TASK_ID' "$template"; then
      pass_msg "$template" "array task variable usage detected"
    else
      fail_msg "$template" "array directive present but SGE_TASK_ID usage missing"
    fi
  fi

  if rg -q '^[[:space:]]*#\$[[:space:]]*-now[[:space:]]+y([[:space:]]|$)' "$template"; then
    fail_msg "$template" "queue fairness violation: -now y is not allowed for batch automation"
  fi

  if rg -q '^[[:space:]]*#\$[[:space:]]*-pe[[:space:]]+omp[[:space:]]+[0-9]+' "$template"; then
    local omp_slots
    omp_slots="$(rg -o --pcre2 '^[[:space:]]*#\$[[:space:]]*-pe[[:space:]]+omp[[:space:]]+\K[0-9]+' "$template" -m 1 || true)"
    if [[ -z "$omp_slots" ]]; then
      fail_msg "$template" "unable to parse omp slots from -pe directive"
    elif ((omp_slots <= 1)); then
      pass_msg "$template" "single-slot omp template does not require thread-slot alignment markers"
    elif rg -q 'NSLOTS|OMP_NUM_THREADS' "$template"; then
      pass_msg "$template" "thread-slot alignment markers found"
    else
      fail_msg "$template" "omp PE requested with slots >1 but NSLOTS or OMP_NUM_THREADS alignment is missing"
    fi

    if [[ "$require_mem_per_core" -eq 1 ]]; then
      if [[ -n "${omp_slots:-}" ]] && ((omp_slots > 1)) && rg -q '^[[:space:]]*#\$[[:space:]]*-l[[:space:]]+mem_per_core=' "$template"; then
        pass_msg "$template" "mem_per_core present for multi-slot omp template"
      elif [[ -n "${omp_slots:-}" ]] && ((omp_slots > 1)); then
        fail_msg "$template" "mem_per_core required for multi-slot omp template"
      else
        pass_msg "$template" "mem_per_core requirement skipped for single-slot omp template"
      fi
    fi
  fi

  if rg -q '/usr[0-9]*/' "$template"; then
    warn_msg "$template" "home-directory-like path detected; prefer /project or /projectnb for production outputs"
  fi

  if rg -q 'gpu|CUDA_VISIBLE_DEVICES' "$template"; then
    pass_msg "$template" "gpu-related directives or environment markers detected"
  fi
}

main() {
  parse_args "$@"

  for template in "${templates[@]}"; do
    check_template "$template"
  done

  printf 'SUMMARY fail=%d warn=%d templates=%d\n' "$fail_count" "$warn_count" "${#templates[@]}"
  if [[ "$fail_count" -gt 0 ]]; then
    exit 1
  fi
}

main "$@"
