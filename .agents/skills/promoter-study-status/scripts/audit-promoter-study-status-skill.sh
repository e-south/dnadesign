#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../../.." && pwd)"
SKILL_FILE="$ROOT_DIR/SKILL.md"
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

require_section() {
  local section="$1"
  if grep -Fxq "$section" "$SKILL_FILE"; then
    pass "section present: $section"
  else
    fail "section missing: $section"
  fi
}

require_pattern() {
  local pattern="$1"
  local label="$2"
  local path="${3:-$SKILL_FILE}"
  if grep -Eq "$pattern" "$path"; then
    pass "$label"
  else
    fail "$label"
  fi
}

require_absent() {
  local pattern="$1"
  local label="$2"
  local path="${3:-$SKILL_FILE}"
  if grep -Eq "$pattern" "$path"; then
    fail "$label"
  else
    pass "$label"
  fi
}

require_file "$SKILL_FILE"
require_file "$REPO_ROOT/docs/studies/README.md"
require_file "$REPO_ROOT/docs/studies/index.yaml"
require_file "$REPO_ROOT/src/dnadesign/usr/docs/operations/promoter-study-status-contract.md"
require_file "$REPO_ROOT/src/dnadesign/usr/docs/operations/promoter-study-preflight.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/ops.study.yaml"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/status.md"

require_section "## Scope"
require_section "## Success Criteria"
require_section "## Workflow"
require_section "## Output"
require_section "## Trigger Tests"

require_pattern 'docs/studies/README\.md' "skill routes through study records docs"
require_pattern 'docs/studies/index\.yaml' "skill requires active study index"
require_pattern 'promoter-study-status-contract\.md' "skill points at status contract"
require_pattern 'promoter-study-preflight' "skill points at preflight route"
require_pattern 'docs/studies/<study-id>/campaign\.yaml' "skill references checked-in campaign manifest"
require_pattern 'docs/studies/<study-id>/datasets\.yaml' "skill references dataset registry"
require_pattern 'docs/studies/<study-id>/status\.md' "skill references status note"
require_pattern 'docs/studies/<study-id>/ops\.study\.yaml' "skill references ops study contract"
require_pattern 'ops progress show usr\.data-plane\.promoter-study-status --json' "skill includes snapshot command"
require_pattern 'ops progress show usr\.data-plane\.promoter-study-preflight --scope next --json' "skill includes next-scope preflight command"
require_pattern 'ops progress campaign --repo-root <repo-root> --manifest docs/studies/<study-id>/campaign\.yaml' "skill includes campaign refresh command"
require_pattern 'usr\.data-plane\.hpc-sync' "skill includes sync audit status route"
require_pattern 'onboard_mode: existing_remote' "skill documents strict remote bootstrap posture"
require_pattern 'source-assembly mode|source/handoff mode' "skill preserves source-phase reporting guidance"
require_pattern 'what should run next\?' "skill differentiates next-step questions"
require_pattern 'ops\.study\.yaml now owns|ops\.study\.yaml' "skill uses checked-in ops study contract"
require_pattern 'strict submit-readiness|strict submit readiness|default notify-enabled Infer presets' "skill documents strict notify-enabled submit readiness"

require_absent 'docs/studies/promoter/' "skill avoids legacy family-nested study paths"
require_absent 'docs/studies/<family>/' "skill avoids placeholder family-nested study paths"

if [[ $failures -eq 0 ]]; then
  printf 'Audit finished with no failures.\n'
else
  printf 'Audit finished with %d failure(s).\n' "$failures"
  exit 1
fi
