#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../../.." && pwd)"
SKILL_FILE="$ROOT_DIR/SKILL.md"
REFERENCE_DIR="$ROOT_DIR/references"
ROOT_AGENTS="$REPO_ROOT/AGENTS.md"
CRUNCHER_AGENTS="$REPO_ROOT/src/dnadesign/cruncher/AGENTS.md"
OLD_SKILL_NAME="snapback""-hairpin-study"
OLD_SKILL_DIR="$REPO_ROOT/.agents/skills/$OLD_SKILL_NAME"
failures=0
SIZE_BUDGET_LINES=180

TMP_COMBINED="$(mktemp)"
trap 'rm -f "$TMP_COMBINED"' EXIT

pass() { printf 'PASS: %s\n' "$1"; }
fail() { printf 'FAIL: %s\n' "$1"; failures=$((failures + 1)); }

require_file() {
  local path="$1"
  [[ -f "$path" ]] && pass "found $path" || fail "missing file $path"
}

require_section() {
  local section="$1"
  grep -Fxq "$section" "$SKILL_FILE" && pass "section present: $section" || fail "section missing: $section"
}

require_pattern() {
  local pattern="$1"
  local label="$2"
  local path="${3:-$TMP_COMBINED}"
  grep -Eq "$pattern" "$path" && pass "$label" || fail "$label"
}

reject_pattern() {
  local pattern="$1"
  local label="$2"
  local path="${3:-$TMP_COMBINED}"
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
  (( lines <= budget )) && pass "SKILL.md line budget respected (${lines} <= ${budget})" || fail "SKILL.md exceeds line budget (${lines} > ${budget})"
}

require_file "$SKILL_FILE"
require_file "$ROOT_AGENTS"
require_file "$CRUNCHER_AGENTS"
require_file "$REPO_ROOT/docs/studies/README.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/status.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/routes.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/scar-nick-base-junction.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/pipeline.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/ops.study.yaml"
require_file "$REPO_ROOT/src/dnadesign/cruncher/docs/operations/cruncher-study-status.md"
require_file "$REPO_ROOT/src/dnadesign/cruncher/docs/operations/cruncher-study-preflight.md"
require_file "$REFERENCE_DIR/route-matrix.md"
require_file "$REFERENCE_DIR/refresh-loop.md"
require_file "$REFERENCE_DIR/study-surfaces.md"
require_file "$REFERENCE_DIR/external-sources.md"

if [[ -e "$OLD_SKILL_DIR" ]]; then
  fail "old skill directory removed"
else
  pass "old skill directory removed"
fi

cat "$SKILL_FILE" > "$TMP_COMBINED"
while IFS= read -r ref; do
  printf '\n' >> "$TMP_COMBINED"
  cat "$ref" >> "$TMP_COMBINED"
done < <(find "$REFERENCE_DIR" -maxdepth 1 -type f -name '*.md' | sort)
{
  printf '\n'
  cat "$ROOT_AGENTS"
  printf '\n'
  cat "$CRUNCHER_AGENTS"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/README.md"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/status.md"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/routes.md"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/pipeline.yaml"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/ops.study.yaml"
} >> "$TMP_COMBINED"

require_section "## Scope"
require_section "## Success Criteria"
require_section "## Workflow"
require_section "## Required Deliverables"
require_section "## Output"
require_section "## Trigger Tests"
require_line_budget "$SKILL_FILE" "$SIZE_BUDGET_LINES"

require_pattern 'docs/studies/README\.md' "skill routes through study records docs"
require_pattern 'docs/studies/retron_hairpin_design/status\.md' "skill references pinned study status"
require_pattern 'docs/studies/retron_hairpin_design/routes\.md' "skill references pinned study routes"
require_pattern 'docs/studies/retron_hairpin_design/scar-nick-base-junction\.md' "skill references scar-nick base-junction context"
require_pattern 'docs/studies/retron_hairpin_design/pipeline\.yaml' "skill references pinned study pipeline"
require_pattern 'cruncher\.data-plane\.cruncher-study-status --study-dir docs/studies/retron_hairpin_design --json' "skill includes pinned snapshot command"
require_pattern 'cruncher\.data-plane\.cruncher-study-preflight --study-dir docs/studies/retron_hairpin_design --scope next --json' "skill includes pinned preflight command"
require_pattern 'released-product Snapback' "skill preserves released-product boundary"
require_pattern 'scar-nick' "skill preserves scar-nick boundary language"
require_pattern 'S0=M' "skill preserves scar-compatible ligation invariant"
require_pattern 'YIU' "skill preserves YIU boundary language"
require_pattern 'harness-engineering' "skill routes harness work outward"
require_pattern 'code-change-discipline' "skill routes contract work outward"
require_pattern '\.agents/skills/retron-hairpin-study/SKILL\.md' "root AGENTS mention the skill" "$ROOT_AGENTS"
require_pattern '\.agents/skills/retron-hairpin-study/SKILL\.md' "cruncher AGENTS mentions the skill" "$CRUNCHER_AGENTS"
reject_pattern "$OLD_SKILL_NAME" "old skill route removed"
require_pattern 'Pair with `harness-engineering`' "skill explains harness pairing" "$SKILL_FILE"
require_pattern 'Pair with `code-change-discipline`' "skill explains code-change pairing" "$SKILL_FILE"

if [[ $failures -eq 0 ]]; then
  printf 'Audit finished with no failures.\n'
else
  printf 'Audit finished with %d failure(s).\n' "$failures"
  exit 1
fi
