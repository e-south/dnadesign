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

  local qstat_source qstat_data
  qstat_source="none"
  if [[ -n "$qstat_file" ]]; then
    qstat_source="fixture"
  elif [[ "$qstat_ok" == "yes" ]]; then
    qstat_source="live"
  fi
  qstat_data="$(read_qstat_data)"

  local total_jobs running_jobs queued_jobs hold_jobs eqw_jobs
  read -r total_jobs running_jobs queued_jobs hold_jobs eqw_jobs <<<"$(summarize_qstat "$qstat_data")"

  local threshold_exceeded
  threshold_exceeded="no"
  if ((running_jobs > warn_over_running)); then
    threshold_exceeded="yes"
  fi

  if [[ "$json_output" -eq 1 ]]; then
    printf '{"host":"%s","user":"%s","cwd":"%s","execution_locus_guess":"%s","scheduler_tools":{"qsub":"%s","qstat":"%s","qdel":"%s"},"job_counts":{"total_jobs":%d,"running_jobs":%d,"queued_jobs":%d,"hold_jobs":%d,"eqw_jobs":%d},"threshold":{"running_threshold":%d,"threshold_exceeded":"%s"},"qstat_source":"%s"}\n' \
      "$(json_escape "$host")" \
      "$(json_escape "$user")" \
      "$(json_escape "$cwd")" \
      "$(json_escape "$locus")" \
      "$qsub_ok" "$qstat_ok" "$qdel_ok" \
      "$total_jobs" "$running_jobs" "$queued_jobs" "$hold_jobs" "$eqw_jobs" \
      "$warn_over_running" "$threshold_exceeded" "$qstat_source"
  else
    printf 'SESSION host=%s user=%s cwd=%s execution_locus_guess=%s\n' "$host" "$user" "$cwd" "$locus"
    printf 'SGE_TOOLS qsub=%s qstat=%s qdel=%s qstat_source=%s\n' "$qsub_ok" "$qstat_ok" "$qdel_ok" "$qstat_source"
    printf 'JOBS total_jobs=%d running_jobs=%d queued_jobs=%d hold_jobs=%d eqw_jobs=%d threshold=%d threshold_exceeded=%s\n' \
      "$total_jobs" "$running_jobs" "$queued_jobs" "$hold_jobs" "$eqw_jobs" "$warn_over_running" "$threshold_exceeded"

    if [[ "$threshold_exceeded" == "yes" ]]; then
      printf 'WARN running_jobs=%d threshold=%d action=confirm-before-additional-submit recommend=array-or-hold_jid\n' \
        "$running_jobs" "$warn_over_running"
    fi

    if ((eqw_jobs > 0)); then
      printf 'WARN eqw_jobs=%d action=triage-before-retry\n' "$eqw_jobs"
    fi
  fi
}

main "$@"
