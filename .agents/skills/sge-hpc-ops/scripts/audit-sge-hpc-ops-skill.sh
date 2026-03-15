#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILL_FILE="$ROOT_DIR/SKILL.md"
GENERIC_AUDIT="$ROOT_DIR/scripts/audit-skill.sh"
FIXTURE_DIR="$ROOT_DIR/references/audit-fixtures"
READ_ONLY_AUDIT="${SKILL_AUDIT_READ_ONLY:-${AGENT_HUB_AUDIT_READ_ONLY:-0}}"
failures=0
temp_files=()

cleanup() {
  if [ "${#temp_files[@]}" -gt 0 ]; then
    rm -f "${temp_files[@]}"
  fi
}

trap cleanup EXIT INT TERM

pass() {
  printf 'PASS: %s\n' "$1"
}

fail() {
  printf 'FAIL: %s\n' "$1"
  failures=$((failures + 1))
}

require_file() {
  local path="$1"
  if [[ -f "$path" ]]; then
    pass "found $(basename "$path")"
  else
    fail "missing file $path"
  fi
}

require_section() {
  local section="$1"
  if grep -Fq "$section" "$SKILL_FILE"; then
    pass "section present: $section"
  else
    fail "section missing: $section"
  fi
}

require_pattern() {
  local pattern="$1"
  local label="$2"
  if rg -q "$pattern" "$SKILL_FILE"; then
    pass "$label"
  else
    fail "$label"
  fi
}

if bash "$GENERIC_AUDIT" "$ROOT_DIR"; then
  pass "generic audit passed"
else
  fail "generic audit failed"
fi

require_file "$SKILL_FILE"
require_file "$ROOT_DIR/references/README.md"
require_file "$ROOT_DIR/references/probe-first-contract.md"
require_file "$ROOT_DIR/references/workflow-router.md"
require_file "$ROOT_DIR/references/route-load-matrix.md"
require_file "$ROOT_DIR/references/interactive-contract.md"
require_file "$ROOT_DIR/references/batch-submit-contract.md"
require_file "$ROOT_DIR/references/automation-qa-preflight.md"
require_file "$ROOT_DIR/references/ci-mechanical-gates.md"
require_file "$ROOT_DIR/references/workload-dnadesign.md"
require_file "$ROOT_DIR/references/bu-scc-system-usage.md"
require_file "$ROOT_DIR/references/source-evidence.md"
require_file "$ROOT_DIR/references/session-status-reporting.md"
require_file "$ROOT_DIR/references/user-status-contract.md"
require_file "$ROOT_DIR/references/submission-shape-advisor.md"
require_file "$ROOT_DIR/references/operator-brief.md"
require_file "$ROOT_DIR/references/test-matrix.md"
require_file "$FIXTURE_DIR/qsub-valid.sh"
require_file "$FIXTURE_DIR/qsub-missing-project.sh"
require_file "$FIXTURE_DIR/qsub-now-y.sh"
require_file "$FIXTURE_DIR/qstat-busy.txt"
require_file "$FIXTURE_DIR/qstat-eqw.txt"
require_file "$FIXTURE_DIR/qstat-green.txt"
require_file "$ROOT_DIR/scripts/qa-sge-submit-preflight.sh"
require_file "$ROOT_DIR/scripts/sge-session-status.sh"
require_file "$ROOT_DIR/scripts/sge-active-jobs.sh"
require_file "$ROOT_DIR/scripts/sge-status-card.sh"
require_file "$ROOT_DIR/scripts/sge-submit-shape-advisor.sh"
require_file "$ROOT_DIR/scripts/sge-operator-brief.sh"

require_section "## Scope"
require_section "## Input Contract"
require_section "## Success Criteria"
require_section "## Workflow"
require_section "## Required Deliverables"
require_section "## Output Contract"
require_section "## Trigger Tests"

require_pattern "Load minimum reference set \(progressive disclosure\)" "progressive disclosure routing step present"
require_pattern "Apply up-to-date handling" "up-to-date handling step present"
require_pattern "verify-before-submit" "verify-before-submit contract present"
require_pattern "qa preflight" "qa preflight contract language present"
require_pattern "workflow_id" "workflow router output key present"
require_pattern "execution_locus" "execution locus output key present"
require_pattern "ondemand_session_handoff" "ondemand handoff route present"
require_pattern "45 days" "freshness threshold in SKILL.md present"
require_pattern "notify" "notify routing language present"
require_pattern "process-reaper" "process-reaper safety language present"
require_pattern "Kubernetes" "non-trigger scheduler boundary present"
require_pattern "more than 3|> 3" "batch concurrency warning threshold present"
require_pattern "session status|session-status" "session status reporting language present"
require_pattern "active[- ]job snapshot|sge-active-jobs.sh" "active-job snapshot language present"
require_pattern "Status card|status card" "user-facing status card language present"
require_pattern "respect the queue|do not skip the line|queue fairness|skip the queue" "queue respect policy language present"
require_pattern "submission-shape advisor|shape advisor" "submission-shape advisor language present"
require_pattern "operator brief|sge-operator-brief.sh" "operator brief language present"

if rg -qi "start a densegen workspace x batch job on bu scc" "$SKILL_FILE"; then
  pass "trigger test includes densegen batch prompt"
else
  fail "missing densegen batch trigger prompt"
fi

if rg -qi "run a densegen workflow .*stress.*ethanol.*cipro.*two hours" "$SKILL_FILE"; then
  pass "trigger test includes stress ethanol cipro two-hour prompt"
else
  fail "missing stress ethanol cipro two-hour trigger prompt"
fi

if rg -qi "also track and wire up notify for slack notifications" "$SKILL_FILE"; then
  pass "trigger test includes notify slack prompt"
else
  fail "missing notify slack trigger prompt"
fi

if rg -qi "submit a request for an interactive on demand session" "$SKILL_FILE"; then
  pass "trigger test includes ondemand request prompt"
else
  fail "missing ondemand request trigger prompt"
fi

if rg -qi "i've just entered into an ondemand session" "$SKILL_FILE"; then
  pass "trigger test includes ondemand handoff prompt"
else
  fail "missing ondemand handoff trigger prompt"
fi

if rg -q "qsub -verify|qsub -w v" "$ROOT_DIR/references/batch-submit-contract.md"; then
  pass "scheduler verification command coverage present"
else
  fail "missing scheduler verification command coverage"
fi

if rg -q "sge-session-status.sh" "$ROOT_DIR/references/batch-submit-contract.md"; then
  pass "batch submit contract includes session status command"
else
  fail "batch submit contract missing session status command"
fi

if rg -q "sge-active-jobs.sh" "$ROOT_DIR/references/batch-submit-contract.md"; then
  pass "batch submit contract includes active-jobs command"
else
  fail "batch submit contract missing active-jobs command"
fi

if rg -q "sge-submit-shape-advisor.sh" "$ROOT_DIR/references/batch-submit-contract.md"; then
  pass "batch submit contract includes shape advisor command"
else
  fail "batch submit contract missing shape advisor command"
fi

if rg -q "respect the queue|do not skip the line|queue fairness" "$ROOT_DIR/references/batch-submit-contract.md"; then
  pass "batch submit contract includes queue respect guidance"
else
  fail "batch submit contract missing queue respect guidance"
fi

if rg -q "Probe 0: execution locus and session context" "$ROOT_DIR/references/probe-first-contract.md"; then
  pass "probe-first contract includes execution-locus probe"
else
  fail "probe-first contract missing execution-locus probe"
fi

if rg -q "Probe D: job activity and queue pressure" "$ROOT_DIR/references/probe-first-contract.md"; then
  pass "probe-first contract includes queue-pressure probe"
else
  fail "probe-first contract missing queue-pressure probe"
fi

if rg -q "OnDemand handoff contract" "$ROOT_DIR/references/interactive-contract.md"; then
  pass "interactive contract includes ondemand handoff guidance"
else
  fail "interactive contract missing ondemand handoff guidance"
fi

if rg -q "DenseGen \+ Notify Slack chained workflow" "$ROOT_DIR/references/workload-dnadesign.md"; then
  pass "dnadesign workload includes chained densegen+notify flow"
else
  fail "dnadesign workload missing chained densegen+notify flow"
fi

if rg -q 'Default packaged solver backend is `CBC`|Solver backend defaults are workspace-specific' "$ROOT_DIR/references/workload-dnadesign.md"; then
  pass "dnadesign workload reference documents solver backend guidance"
else
  fail "dnadesign workload reference missing solver backend guidance"
fi

if rg -q "pquota" "$ROOT_DIR/references/workload-dnadesign.md"; then
  pass "dnadesign workload includes storage quota precheck"
else
  fail "dnadesign workload missing storage quota precheck"
fi

if rg -q "SGE_TASK_ID" "$ROOT_DIR/references/batch-submit-contract.md"; then
  pass "batch submit contract includes array task-id guidance"
else
  fail "batch submit contract missing array task-id guidance"
fi

if rg -q 'OMP_NUM_THREADS=\$NSLOTS' "$ROOT_DIR/references/batch-submit-contract.md"; then
  pass "batch submit contract includes thread-slot alignment guidance"
else
  fail "batch submit contract missing thread-slot alignment guidance"
fi

if rg -q -- "-hold_jid" "$ROOT_DIR/references/batch-submit-contract.md"; then
  pass "batch submit contract includes dependency sequencing guidance"
else
  fail "batch submit contract missing dependency sequencing guidance"
fi

if rg -q "running_jobs > 3|more than 3 running" "$ROOT_DIR/references/automation-qa-preflight.md"; then
  pass "qa preflight includes running-jobs pressure threshold"
else
  fail "qa preflight missing running-jobs pressure threshold"
fi

if rg -q "route-load-matrix.md" "$ROOT_DIR/references/README.md"; then
  pass "reference index includes route load matrix"
else
  fail "reference index missing route load matrix"
fi

if rg -q "session-status-reporting.md" "$ROOT_DIR/references/README.md"; then
  pass "reference index includes session status guide"
else
  fail "reference index missing session status guide"
fi

if rg -q "user-status-contract.md" "$ROOT_DIR/references/README.md"; then
  pass "reference index includes user status contract"
else
  fail "reference index missing user status contract"
fi

if rg -q "submission-shape-advisor.md" "$ROOT_DIR/references/README.md"; then
  pass "reference index includes submission-shape advisor guide"
else
  fail "reference index missing submission-shape advisor guide"
fi

if rg -q "operator-brief.md" "$ROOT_DIR/references/README.md"; then
  pass "reference index includes operator brief guide"
else
  fail "reference index missing operator brief guide"
fi

if rg -q "notify by default|default route .*notify|notify-on by default" "$ROOT_DIR/references/workflow-router.md"; then
  pass "workflow router documents notify-default route policy"
else
  fail "workflow router missing notify-default route policy"
fi

if rg -q -- "--no-notify|without notify|notify off|notify disabled" "$ROOT_DIR/references/workflow-router.md"; then
  pass "workflow router documents explicit notify opt-out cues"
else
  fail "workflow router missing explicit notify opt-out cues"
fi

if rg -q "ops runbook precedents" "$ROOT_DIR/SKILL.md"; then
  pass "skill docs include ops runbook precedents entrypoint"
else
  fail "skill docs missing ops runbook precedents entrypoint"
fi

if rg -q "DenseGen scaffolds include notify by default|--no-notify" "$ROOT_DIR/references/workload-dnadesign.md"; then
  pass "dnadesign workload reference aligns notify default guidance"
else
  fail "dnadesign workload reference missing notify default guidance"
fi

for workflow_id in densegen_batch_submit densegen_batch_with_notify_slack ondemand_session_request ondemand_session_handoff generic_sge_ops; do
  if rg -q "$workflow_id" "$ROOT_DIR/references/workflow-router.md"; then
    pass "workflow router includes $workflow_id"
  else
    fail "workflow router missing $workflow_id"
  fi
done

if "$ROOT_DIR/scripts/qa-sge-submit-preflight.sh" --help >/dev/null 2>&1; then
  pass "qa submit preflight script has working --help"
else
  fail "qa submit preflight script --help failed"
fi

if [ "$READ_ONLY_AUDIT" = "1" ]; then
  tmp_good="$FIXTURE_DIR/qsub-valid.sh"
else
  tmp_good="$(mktemp)"
  temp_files+=("$tmp_good")
  cat >"$tmp_good" <<'QSUB'
#!/bin/bash -l
#$ -P my_project
#$ -l h_rt=02:00:00
#$ -pe omp 4
export OMP_NUM_THREADS=$NSLOTS
echo "ok"
QSUB
fi

if "$ROOT_DIR/scripts/qa-sge-submit-preflight.sh" --template "$tmp_good" --require-project-flag >/dev/null 2>&1; then
  pass "qa submit preflight passes valid template"
else
  fail "qa submit preflight failed valid template"
fi

if [ "$READ_ONLY_AUDIT" = "1" ]; then
  tmp_bad="$FIXTURE_DIR/qsub-missing-project.sh"
else
  tmp_bad="$(mktemp)"
  temp_files+=("$tmp_bad")
  cat >"$tmp_bad" <<'QSUB'
#!/bin/bash
#$ -l h_rt=02:00:00
module load python3/3.13.8
echo "bad"
QSUB
fi

if "$ROOT_DIR/scripts/qa-sge-submit-preflight.sh" --template "$tmp_bad" --require-project-flag >/dev/null 2>&1; then
  fail "qa submit preflight passed invalid template"
else
  pass "qa submit preflight rejects invalid template"
fi

if [ "$READ_ONLY_AUDIT" = "1" ]; then
  tmp_nowy="$FIXTURE_DIR/qsub-now-y.sh"
else
  tmp_nowy="$(mktemp)"
  temp_files+=("$tmp_nowy")
  cat >"$tmp_nowy" <<'QSUB'
#!/bin/bash -l
#$ -P my_project
#$ -l h_rt=01:00:00
#$ -now y
echo "queue bypass risk"
QSUB
fi

if "$ROOT_DIR/scripts/qa-sge-submit-preflight.sh" --template "$tmp_nowy" --require-project-flag >/dev/null 2>&1; then
  fail "qa submit preflight passed queue-bypass template"
else
  pass "qa submit preflight rejects queue-bypass template"
fi

if "$ROOT_DIR/scripts/sge-session-status.sh" --help >/dev/null 2>&1; then
  pass "session status script has working --help"
else
  fail "session status script --help failed"
fi

if "$ROOT_DIR/scripts/sge-active-jobs.sh" --help >/dev/null 2>&1; then
  pass "active-jobs script has working --help"
else
  fail "active-jobs script --help failed"
fi

if [ "$READ_ONLY_AUDIT" = "1" ]; then
  tmp_qstat="$FIXTURE_DIR/qstat-busy.txt"
else
  tmp_qstat="$(mktemp)"
  temp_files+=("$tmp_qstat")
  cat >"$tmp_qstat" <<'QSTAT'
job-ID  prior   name       user         state submit/start at     queue                          slots ja-task-ID
---------------------------------------------------------------------------------------------------------------
1001    0.00000 jobA       testuser     r     02/28/2026 10:00:00 test.q@node1                  1
1002    0.00000 jobB       testuser     r     02/28/2026 10:01:00 test.q@node1                  1
1003    0.00000 jobC       testuser     qw    02/28/2026 10:02:00                               1
1004    0.00000 jobD       testuser     r     02/28/2026 10:03:00 test.q@node2                  1
1005    0.00000 jobE       testuser     r     02/28/2026 10:04:00 test.q@node2                  1
QSTAT
fi

status_output=""
if status_output="$("$ROOT_DIR"/scripts/sge-session-status.sh --qstat-file "$tmp_qstat" --warn-over-running 3 2>/dev/null)"; then
  if printf '%s\n' "$status_output" | rg -q "running_jobs=4"; then
    pass "session status script reports running job count"
  else
    fail "session status script missing running job count"
  fi
  if printf '%s\n' "$status_output" | rg -q "WARN .*threshold=3"; then
    pass "session status script warns over running threshold"
  else
    fail "session status script missing over-threshold warning"
  fi
else
  fail "session status script failed fixture run"
fi

active_jobs_output=""
if active_jobs_output="$("$ROOT_DIR"/scripts/sge-active-jobs.sh --qstat-file "$tmp_qstat" --max-jobs 2 2>/dev/null)"; then
  if printf '%s\n' "$active_jobs_output" | rg -q "ACTIVE_JOBS total_jobs=5 shown_jobs=2"; then
    pass "active-jobs script reports total and shown counts"
  else
    fail "active-jobs script missing count summary"
  fi
  if printf '%s\n' "$active_jobs_output" | rg -q "^1001[[:space:]]+jobA[[:space:]]+r"; then
    pass "active-jobs script reports job id and state rows"
  else
    fail "active-jobs script missing expected row data"
  fi
else
  fail "active-jobs script failed fixture run"
fi

active_jobs_json=""
if active_jobs_json="$("$ROOT_DIR"/scripts/sge-active-jobs.sh --qstat-file "$tmp_qstat" --max-jobs 2 --json 2>/dev/null)"; then
  if printf '%s\n' "$active_jobs_json" | rg -q '"total_jobs":5'; then
    pass "active-jobs script json reports total_jobs"
  else
    fail "active-jobs script json missing total_jobs"
  fi
  if printf '%s\n' "$active_jobs_json" | rg -q '"job_id":"1001"'; then
    pass "active-jobs script json includes job id"
  else
    fail "active-jobs script json missing job id"
  fi
else
  fail "active-jobs script failed json fixture run"
fi

if "$ROOT_DIR/scripts/sge-status-card.sh" --help >/dev/null 2>&1; then
  pass "status card script has working --help"
else
  fail "status card script --help failed"
fi

card_output=""
if card_output="$("$ROOT_DIR"/scripts/sge-status-card.sh --qstat-file "$tmp_qstat" --warn-over-running 3 2>/dev/null)"; then
  if printf '%s\n' "$card_output" | rg -q "Health: yellow"; then
    pass "status card script reports yellow health over threshold"
  else
    fail "status card script missing yellow health status"
  fi
  if printf '%s\n' "$card_output" | rg -q "Recommendation: Confirm before additional submissions"; then
    pass "status card script includes confirmation recommendation"
  else
    fail "status card script missing confirmation recommendation"
  fi
else
  fail "status card script failed fixture run"
fi

if [ "$READ_ONLY_AUDIT" = "1" ]; then
  tmp_qstat_eqw="$FIXTURE_DIR/qstat-eqw.txt"
else
  tmp_qstat_eqw="$(mktemp)"
  temp_files+=("$tmp_qstat_eqw")
  cat >"$tmp_qstat_eqw" <<'QSTAT'
job-ID  prior   name       user         state submit/start at     queue                          slots ja-task-ID
---------------------------------------------------------------------------------------------------------------
2001    0.00000 jobErr     testuser     Eqw   02/28/2026 11:00:00 test.q@node1                  1
QSTAT
fi

eqw_card_output=""
if eqw_card_output="$("$ROOT_DIR"/scripts/sge-status-card.sh --qstat-file "$tmp_qstat_eqw" --warn-over-running 3 2>/dev/null)"; then
  if printf '%s\n' "$eqw_card_output" | rg -q "Health: red"; then
    pass "status card script reports red health for Eqw"
  else
    fail "status card script missing red health for Eqw"
  fi
  if printf '%s\n' "$eqw_card_output" | rg -q "Recommendation: Triage Eqw jobs before additional submissions"; then
    pass "status card script includes Eqw triage recommendation"
  else
    fail "status card script missing Eqw triage recommendation"
  fi
else
  fail "status card script failed Eqw fixture run"
fi

if "$ROOT_DIR/scripts/sge-submit-shape-advisor.sh" --help >/dev/null 2>&1; then
  pass "shape advisor script has working --help"
else
  fail "shape advisor script --help failed"
fi

if "$ROOT_DIR/scripts/sge-operator-brief.sh" --help >/dev/null 2>&1; then
  pass "operator brief script has working --help"
else
  fail "operator brief script --help failed"
fi

advisor_output=""
if advisor_output="$("$ROOT_DIR"/scripts/sge-submit-shape-advisor.sh --qstat-file "$tmp_qstat" --planned-submits 8 --warn-over-running 3 2>/dev/null)"; then
  if printf '%s\n' "$advisor_output" | rg -q "advisor=array"; then
    pass "shape advisor recommends array when over threshold"
  else
    fail "shape advisor missing array recommendation"
  fi
else
  fail "shape advisor failed array fixture run"
fi

advisor_chain_output=""
if advisor_chain_output="$("$ROOT_DIR"/scripts/sge-submit-shape-advisor.sh --qstat-file "$tmp_qstat" --planned-submits 8 --requires-order --warn-over-running 3 2>/dev/null)"; then
  if printf '%s\n' "$advisor_chain_output" | rg -q "advisor=hold_jid"; then
    pass "shape advisor recommends hold_jid for ordered workload"
  else
    fail "shape advisor missing hold_jid recommendation"
  fi
else
  fail "shape advisor failed hold_jid fixture run"
fi

brief_high_output=""
if brief_high_output="$("$ROOT_DIR"/scripts/sge-operator-brief.sh --qstat-file "$tmp_qstat" --planned-submits 8 --warn-over-running 3 2>/dev/null)"; then
  if printf '%s\n' "$brief_high_output" | rg -q "Submit Gate: confirm"; then
    pass "operator brief reports confirmation gate under high pressure"
  else
    fail "operator brief missing confirmation gate under high pressure"
  fi
  if printf '%s\n' "$brief_high_output" | rg -q "Advisor: array"; then
    pass "operator brief includes array advisor under high pressure"
  else
    fail "operator brief missing array advisor under high pressure"
  fi
else
  fail "operator brief failed high-pressure fixture run"
fi

brief_high_json=""
if brief_high_json="$("$ROOT_DIR"/scripts/sge-operator-brief.sh --qstat-file "$tmp_qstat" --planned-submits 8 --warn-over-running 3 --json 2>/dev/null)"; then
  if printf '%s\n' "$brief_high_json" | rg -q '"submit_gate":"confirm"'; then
    pass "operator brief json reports confirmation gate under high pressure"
  else
    fail "operator brief json missing confirmation gate under high pressure"
  fi
  if printf '%s\n' "$brief_high_json" | rg -q '"running_jobs":4'; then
    pass "operator brief json includes numeric running job count"
  else
    fail "operator brief json missing numeric running job count"
  fi
else
  fail "operator brief failed high-pressure json fixture run"
fi

brief_eqw_output=""
if brief_eqw_output="$("$ROOT_DIR"/scripts/sge-operator-brief.sh --qstat-file "$tmp_qstat_eqw" --planned-submits 2 --warn-over-running 3 2>/dev/null)"; then
  if printf '%s\n' "$brief_eqw_output" | rg -q "Submit Gate: block"; then
    pass "operator brief reports block gate for Eqw"
  else
    fail "operator brief missing block gate for Eqw"
  fi
else
  fail "operator brief failed Eqw fixture run"
fi

if [ "$READ_ONLY_AUDIT" = "1" ]; then
  tmp_qstat_green="$FIXTURE_DIR/qstat-green.txt"
else
  tmp_qstat_green="$(mktemp)"
  temp_files+=("$tmp_qstat_green")
  cat >"$tmp_qstat_green" <<'QSTAT'
job-ID  prior   name       user         state submit/start at     queue                          slots ja-task-ID
---------------------------------------------------------------------------------------------------------------
3001    0.00000 jobA       testuser     r     02/28/2026 12:00:00 test.q@node1                  1
3002    0.00000 jobB       testuser     qw    02/28/2026 12:01:00                               1
QSTAT
fi

brief_green_output=""
if brief_green_output="$("$ROOT_DIR"/scripts/sge-operator-brief.sh --qstat-file "$tmp_qstat_green" --planned-submits 1 --warn-over-running 3 2>/dev/null)"; then
  if printf '%s\n' "$brief_green_output" | rg -q "Submit Gate: ready"; then
    pass "operator brief reports ready gate under normal pressure"
  else
    fail "operator brief missing ready gate under normal pressure"
  fi
else
  fail "operator brief failed normal-pressure fixture run"
fi

required_urls=(
  "https://www.bu.edu/tech/support/research/system-usage/"
  "https://www.bu.edu/tech/support/research/system-usage/connect-scc/"
  "https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/"
  "https://www.bu.edu/tech/support/research/system-usage/scc-environment/"
  "https://www.bu.edu/tech/support/research/system-usage/transferring-files/"
  "https://www.bu.edu/tech/support/research/system-usage/running-jobs/"
  "https://www.bu.edu/tech/support/research/system-usage/running-jobs/best-practices/"
  "https://www.bu.edu/tech/support/research/system-usage/using-file-system/file-system-structure/"
  "https://www.bu.edu/tech/support/research/system-usage/running-jobs/advanced-batch/"
  "https://www.bu.edu/tech/support/research/system-usage/running-jobs/tracking-jobs/"
  "https://www.bu.edu/tech/support/research/system-usage/running-jobs/parallel-batch/"
  "https://www.bu.edu/tech/support/research/system-usage/running-jobs/allocating-memory-for-your-job/"
  "https://www.bu.edu/tech/support/research/system-usage/running-jobs/resources-jobs/"
  "https://www.bu.edu/tech/support/research/system-usage/running-jobs/batch-script-examples/"
  "https://www.bu.edu/tech/support/research/system-usage/running-jobs/process-reaper/"
)

for url in "${required_urls[@]}"; do
  if rg -Fq "$url" "$ROOT_DIR/references/source-evidence.md"; then
    pass "source evidence includes $url"
  else
    fail "source evidence missing $url"
  fi

  if rg -Fq "$url" "$ROOT_DIR/references/bu-scc-system-usage.md"; then
    pass "system usage map includes $url"
  else
    fail "system usage map missing $url"
  fi
done

for claim in "5 interactive jobs" "10 GB" "75,000" "scc-globus.bu.edu" "15 min"; do
  if rg -Fq "$claim" "$ROOT_DIR/references/bu-scc-system-usage.md" "$ROOT_DIR/references/source-evidence.md"; then
    pass "volatile claim coverage includes: $claim"
  else
    fail "volatile claim coverage missing: $claim"
  fi
done

if python3 - "$ROOT_DIR/references/source-evidence.md" <<'PY'; then
import datetime
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
if not dates:
    print("no retrieval dates found")
    raise SystemExit(1)

max_age_days = 45
today = datetime.date.today()
errors = []
for raw in sorted(set(dates)):
    try:
        parsed = datetime.datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        errors.append(f"invalid date format: {raw}")
        continue
    age = (today - parsed).days
    if age < 0:
        errors.append(f"retrieval date in future: {raw}")
    elif age > max_age_days:
        errors.append(f"retrieval date stale by {age} days: {raw}")

if errors:
    for err in errors:
        print(err)
    raise SystemExit(1)
PY
  pass "source evidence retrieval dates are valid and within 45 days"
else
  fail "source evidence retrieval date freshness check failed"
fi

line_count="$(wc -l <"$SKILL_FILE" | tr -d ' ')"
if [[ "$line_count" -le 240 ]]; then
  pass "SKILL.md line budget is within 240 lines"
else
  fail "SKILL.md exceeds 240 line budget (found $line_count)"
fi

if [[ $failures -gt 0 ]]; then
  printf 'Audit finished with %d failure(s).\n' "$failures"
  exit 1
fi

echo "Audit finished with no failures."
