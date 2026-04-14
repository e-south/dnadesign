#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../../.." && pwd)"
SKILL_FILE="$ROOT_DIR/SKILL.md"
REFERENCE_DIR="$ROOT_DIR/references"
failures=0
SIZE_BUDGET_LINES=170

TMP_COMBINED="$(mktemp)"
trap 'rm -f "$TMP_COMBINED"' EXIT

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
  local path="${3:-$TMP_COMBINED}"
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

require_line_budget() {
  local path="$1"
  local budget="$2"
  local lines
  lines="$(wc -l < "$path" | tr -d ' ')"
  if (( lines <= budget )); then
    pass "SKILL.md line budget respected (${lines} <= ${budget})"
  else
    fail "SKILL.md exceeds line budget (${lines} > ${budget})"
  fi
}

require_reference_pack() {
  local count
  count="$(find "$REFERENCE_DIR" -maxdepth 1 -type f -name '*.md' | wc -l | tr -d ' ')"
  if (( count >= 2 )); then
    pass "reference pack present (${count} markdown files)"
  else
    fail "reference pack must contain at least two markdown files"
  fi
}

require_file "$SKILL_FILE"
require_file "$REPO_ROOT/docs/studies/README.md"
require_file "$REPO_ROOT/docs/studies/index.yaml"
require_file "$REPO_ROOT/src/dnadesign/usr/docs/operations/promoter-study-status-contract.md"
require_file "$REPO_ROOT/src/dnadesign/usr/docs/operations/promoter-study-preflight.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/ops.study.yaml"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/status.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes.md"
require_file "$REFERENCE_DIR/external-sources.md"
require_reference_pack

cat "$SKILL_FILE" > "$TMP_COMBINED"
while IFS= read -r ref; do
  printf '\n' >> "$TMP_COMBINED"
  cat "$ref" >> "$TMP_COMBINED"
done < <(find "$REFERENCE_DIR" -maxdepth 1 -type f -name '*.md' | sort)

require_section "## Scope"
require_section "## Success Criteria"
require_section "## Workflow"
require_section "## Required Deliverables"
require_section "## Output"
require_section "## Trigger Tests"
require_line_budget "$SKILL_FILE" "$SIZE_BUDGET_LINES"

require_pattern 'references/external-sources\.md' "skill exposes external sources reference" "$SKILL_FILE"
require_pattern 'references/route-matrix\.md' "skill exposes route matrix reference" "$SKILL_FILE"
require_pattern 'references/refresh-loop\.md' "skill exposes refresh loop reference" "$SKILL_FILE"
require_pattern 'references/study-surfaces\.md' "skill exposes study surfaces reference" "$SKILL_FILE"

require_pattern 'docs/studies/README\.md' "skill routes through study records docs"
require_pattern 'docs/studies/index\.yaml' "skill requires active study index"
require_pattern 'promoter-study-status-contract\.md' "skill points at status contract"
require_pattern 'promoter-study-preflight' "skill points at preflight route"
require_pattern 'docs/studies/<study-id>/campaign\.yaml' "skill references checked-in campaign manifest"
require_pattern 'docs/studies/<study-id>/datasets\.yaml' "skill references dataset registry"
require_pattern 'docs/studies/<study-id>/status\.md' "skill references status note"
require_pattern 'docs/studies/<study-id>/ops\.study\.yaml' "skill references ops study contract"
require_pattern 'docs/studies/<study-id>/routes\.md' "skill references study-owned route map"
require_pattern 'ops progress show usr\.data-plane\.promoter-study-status --json' "skill includes snapshot command"
require_pattern 'ops progress show usr\.data-plane\.promoter-study-preflight --scope next --json' "skill includes next-scope preflight command"
require_pattern 'ops progress campaign --repo-root <repo-root> --manifest docs/studies/<study-id>/campaign\.yaml' "skill includes campaign refresh command"
require_pattern 'usr\.data-plane\.hpc-sync' "skill includes sync audit status route"
require_pattern 'onboard_mode: existing_remote' "skill documents strict remote bootstrap posture"
require_pattern 'source/handoff mode' "skill preserves source-phase reporting guidance"
require_pattern 'what blocks execution here\?' "skill differentiates blocker questions"
require_pattern 'ops\.study\.yaml' "skill uses checked-in ops study contract"
require_pattern 'default notify-enabled Infer presets|default notify-enabled infer presets' "skill documents strict notify-enabled submit readiness"

require_absent 'docs/studies/promoter/' "skill avoids legacy family-nested study paths" "$TMP_COMBINED"
require_absent 'docs/studies/<family>/' "skill avoids placeholder family-nested study paths" "$TMP_COMBINED"
require_absent 'Refresh affiliated-dataset sync posture' "skill avoids inline sync walkthrough section"
require_absent 'Refresh pending infer slices' "skill avoids inline infer walkthrough section"
require_absent 'Refresh batch or notify evidence only when asked' "skill avoids inline notify walkthrough section"
require_absent 'notify setup resolve-events' "skill top level avoids inline notify command walkthrough"
require_absent 'usr --root <usr-root> diff <dataset-id> <remote-name>' "skill top level avoids inline sync command walkthrough"
require_absent 'infer run --config <infer-config> --dry-run' "skill top level avoids inline infer dry-run walkthrough"
require_absent 'usr maintenance overlay-remove|infer prune' "skill top level avoids rollback command inventory"

if [[ $failures -eq 0 ]]; then
  printf 'Audit finished with no failures.\n'
else
  printf 'Audit finished with %d failure(s).\n' "$failures"
  exit 1
fi
