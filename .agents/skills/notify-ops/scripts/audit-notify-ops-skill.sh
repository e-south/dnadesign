#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../../.." && pwd)"
SKILL_FILE="$ROOT_DIR/SKILL.md"
REFERENCE_DIR="$ROOT_DIR/references"
ROOT_AGENTS="$REPO_ROOT/AGENTS.md"
NOTIFY_AGENTS="$REPO_ROOT/src/dnadesign/notify/AGENTS.md"
failures=0

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
    pass "found $(realpath --relative-to="$REPO_ROOT" "$path" 2>/dev/null || echo "$path")"
  else
    fail "missing file $path"
  fi
}

require_text() {
  local needle="$1"
  local label="$2"
  local path="${3:-$SKILL_FILE}"
  if grep -Fq -- "$needle" "$path"; then
    pass "$label"
  else
    fail "$label"
  fi
}

require_absent() {
  local needle="$1"
  local label="$2"
  local path="${3:-$NOTIFY_AGENTS}"
  if grep -Fq -- "$needle" "$path"; then
    fail "$label"
  else
    pass "$label"
  fi
}

require_file "$SKILL_FILE"
require_file "$REFERENCE_DIR/workflow-router.md"
require_file "$REFERENCE_DIR/external-sources.md"
require_file "$ROOT_AGENTS"
require_file "$NOTIFY_AGENTS"
require_file "$REPO_ROOT/docs/notify/README.md"
require_file "$REPO_ROOT/docs/notify/usr-events.md"
require_file "$REPO_ROOT/src/dnadesign/notify/docs/reference/command-contracts.md"

for section in "## Scope" "## Success Criteria" "## Workflow" "## Required Deliverables" "## Output" "## Trigger Tests"; do
  require_text "$section" "section present: $section"
done

require_text ".agents/skills/notify-ops/SKILL.md" "root AGENTS routes to notify skill" "$ROOT_AGENTS"
require_text ".agents/skills/notify-ops/SKILL.md" "notify AGENTS routes to notify skill" "$NOTIFY_AGENTS"
require_text "docs/notify/README.md" "notify AGENTS links operator overview" "$NOTIFY_AGENTS"
require_text "docs/notify/usr-events.md" "notify AGENTS links operator runbook" "$NOTIFY_AGENTS"
require_text "src/dnadesign/notify/docs/reference/command-contracts.md" "notify AGENTS links command contracts" "$NOTIFY_AGENTS"
require_text "docs/notify/README.md" "skill links operator overview"
require_text "docs/notify/usr-events.md" "skill links operator runbook"
require_text "src/dnadesign/notify/docs/reference/command-contracts.md" "skill links command contracts"
require_text "docs/bu-scc/batch-notify.md" "skill routes scheduler-backed watcher work"
require_text ".events.log" "skill keeps USR events boundary explicit"
require_text "outputs/meta/events.jsonl" "skill rejects DenseGen runtime telemetry as input"
require_text "--secret-source file --secret-ref file://" "skill prefers file-backed webhook refs"

require_absent "Default low-friction flow" "notify AGENTS avoids inline operator flow"
require_absent "notify setup slack" "notify AGENTS avoids inline setup commands"
require_absent "notify usr-events watch --tool <tool> --workspace <workspace-name> --follow" "notify AGENTS avoids inline watch commands"
require_absent "--secret-source auto" "notify AGENTS avoids stale secret-source guidance"

if [[ $failures -eq 0 ]]; then
  printf 'Audit finished with no failures.\n'
else
  printf 'Audit finished with %d failure(s).\n' "$failures"
  exit 1
fi
