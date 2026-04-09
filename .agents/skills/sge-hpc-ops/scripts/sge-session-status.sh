#!/usr/bin/env bash
set -euo pipefail

warn_over_running=3
qstat_file=""
json_output=0

usage() {
  cat <<'USAGE'
Usage:
  sge-session-status.sh [options]

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

is_submit_host_denied() {
  local message
  message="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  [[ "$message" == *"is no submit host"* || "$message" == *"neither submit nor admin host"* ]]
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

detect_locus() {
  local host="$1"
  if [[ -n "${OOD_PORTAL:-}" || -n "${OOD_SESSION_TOKEN:-}" ]]; then
    echo "ondemand_shell"
    return
  fi

  if [[ "$host" == *"scc"* || "$host" == *"scc-"* || "$host" == *".scc.bu.edu"* ]]; then
    if [[ -n "${SSH_CONNECTION:-}" ]]; then
      echo "scc_login_shell"
    else
      echo "ondemand_app_shell"
    fi
    return
  fi

  if [[ -n "${SSH_CONNECTION:-}" ]]; then
    echo "unknown"
    return
  fi

  echo "local_shell"
}

probe_qstat() {
  queue_probe="ok"
  qstat_source="none"
  qstat_data=""
  probe_error=""

  if [[ -n "$qstat_file" ]]; then
    qstat_source="fixture"
    qstat_data="$(cat "$qstat_file")"
    return
  fi

  if ! command -v qstat >/dev/null 2>&1; then
    queue_probe="degraded"
    probe_error="qstat unavailable"
    return
  fi

  qstat_source="live"
  local output status
  set +e
  output="$(qstat -u "${USER:-$(whoami)}" 2>&1)"
  status=$?
  set -e
  if ((status != 0)); then
    qstat_source="degraded"
    probe_error="$(printf '%s' "$output" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//')"
    if is_submit_host_denied "$probe_error"; then
      queue_probe="host_denied"
    else
      queue_probe="degraded"
    fi
    return
  fi

  qstat_data="$output"
}

summarize_qstat() {
  local data="$1"
  if [[ -z "${data//[[:space:]]/}" ]]; then
    printf '0 0 0 0 0\n'
    return
  fi

  awk '
    BEGIN { total=0; running=0; queued=0; hold=0; eqw=0 }
    $1 ~ /^[0-9]+$/ {
      total++
      state=$5
      if (state ~ /r/) running++
      if (state ~ /q/) queued++
      if (state ~ /h/) hold++
      if (state ~ /Eqw/) eqw++
    }
    END { printf "%d %d %d %d %d\n", total, running, queued, hold, eqw }
  ' <<<"$data"
}

main() {
  parse_args "$@"

  local host user cwd locus
  host="$(hostname 2>/dev/null || echo unknown-host)"
  user="${USER:-$(whoami 2>/dev/null || echo unknown-user)}"
  cwd="$(pwd 2>/dev/null || echo unknown-cwd)"
  locus="$(detect_locus "$host")"

  local qsub_ok qstat_ok qdel_ok
  qsub_ok="no"
  qstat_ok="no"
  qdel_ok="no"
  command -v qsub >/dev/null 2>&1 && qsub_ok="yes"
  command -v qstat >/dev/null 2>&1 && qstat_ok="yes"
  command -v qdel >/dev/null 2>&1 && qdel_ok="yes"

  local queue_probe qstat_source qstat_data probe_error
  probe_qstat

  local total_jobs running_jobs queued_jobs hold_jobs eqw_jobs
  if [[ "$queue_probe" == "ok" ]]; then
    read -r total_jobs running_jobs queued_jobs hold_jobs eqw_jobs <<<"$(summarize_qstat "$qstat_data")"
  else
    total_jobs="unknown"
    running_jobs="unknown"
    queued_jobs="unknown"
    hold_jobs="unknown"
    eqw_jobs="unknown"
  fi

  local threshold_exceeded
  threshold_exceeded="unknown"
  if [[ "$running_jobs" =~ ^[0-9]+$ ]]; then
    threshold_exceeded="no"
  fi
  if [[ "$running_jobs" =~ ^[0-9]+$ ]] && ((running_jobs > warn_over_running)); then
    threshold_exceeded="yes"
  fi

  if [[ "$json_output" -eq 1 ]]; then
    printf '{"host":"%s","user":"%s","cwd":"%s","execution_locus_guess":"%s","scheduler_tools":{"qsub":"%s","qstat":"%s","qdel":"%s"},"job_counts":{"total_jobs":%s,"running_jobs":%s,"queued_jobs":%s,"hold_jobs":%s,"eqw_jobs":%s},"threshold":{"running_threshold":%d,"threshold_exceeded":"%s"},"queue_probe":"%s","qstat_source":"%s","probe_error":"%s"}\n' \
      "$(json_escape "$host")" \
      "$(json_escape "$user")" \
      "$(json_escape "$cwd")" \
      "$(json_escape "$locus")" \
      "$qsub_ok" "$qstat_ok" "$qdel_ok" \
      "$(json_int_or_null "$total_jobs")" \
      "$(json_int_or_null "$running_jobs")" \
      "$(json_int_or_null "$queued_jobs")" \
      "$(json_int_or_null "$hold_jobs")" \
      "$(json_int_or_null "$eqw_jobs")" \
      "$warn_over_running" "$threshold_exceeded" "$queue_probe" "$qstat_source" "$(json_escape "$probe_error")"
  else
    printf 'SESSION host=%s user=%s cwd=%s execution_locus_guess=%s\n' "$host" "$user" "$cwd" "$locus"
    printf 'SGE_TOOLS qsub=%s qstat=%s qdel=%s qstat_source=%s queue_probe=%s\n' \
      "$qsub_ok" "$qstat_ok" "$qdel_ok" "$qstat_source" "$queue_probe"
    printf 'JOBS total_jobs=%s running_jobs=%s queued_jobs=%s hold_jobs=%s eqw_jobs=%s threshold=%d threshold_exceeded=%s\n' \
      "$total_jobs" "$running_jobs" "$queued_jobs" "$hold_jobs" "$eqw_jobs" "$warn_over_running" "$threshold_exceeded"

    if [[ "$threshold_exceeded" == "yes" ]]; then
      printf 'WARN running_jobs=%s threshold=%d action=confirm-before-additional-submit recommend=array-or-hold_jid\n' \
        "$running_jobs" "$warn_over_running"
    fi

    if [[ "$eqw_jobs" =~ ^[0-9]+$ ]] && ((eqw_jobs > 0)); then
      printf 'WARN eqw_jobs=%s action=triage-before-retry\n' "$eqw_jobs"
    fi

    if [[ "$queue_probe" == "host_denied" ]]; then
      printf 'WARN queue_probe=host_denied action=use-submit-host reason=current-host-not-submit-capable\n'
    elif [[ "$queue_probe" == "degraded" ]]; then
      printf 'WARN queue_probe=degraded action=reprobe reason=qstat-unavailable\n'
    fi

    if [[ -n "$probe_error" ]]; then
      printf 'WARN probe_error=%s\n' "$probe_error"
    fi
  fi
}

main "$@"
