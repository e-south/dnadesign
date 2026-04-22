#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../../.." && pwd)"
SKILL_FILE="$ROOT_DIR/SKILL.md"
REFERENCE_DIR="$ROOT_DIR/references"
ROOT_AGENTS="$REPO_ROOT/AGENTS.md"
CRUNCHER_AGENTS="$REPO_ROOT/src/dnadesign/cruncher/AGENTS.md"
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
require_file "$REPO_ROOT/docs/studies/snapback_shortening_effort/status.md"
require_file "$REPO_ROOT/docs/studies/snapback_shortening_effort/routes.md"
require_file "$REPO_ROOT/docs/studies/snapback_shortening_effort/pipeline.yaml"
require_file "$REPO_ROOT/docs/studies/snapback_shortening_effort/ops.study.yaml"
require_file "$REPO_ROOT/src/dnadesign/cruncher/docs/operations/cruncher-study-status.md"
require_file "$REPO_ROOT/src/dnadesign/cruncher/docs/operations/cruncher-study-preflight.md"
require_file "$REFERENCE_DIR/route-matrix.md"
require_file "$REFERENCE_DIR/refresh-loop.md"
require_file "$REFERENCE_DIR/study-surfaces.md"
require_file "$REFERENCE_DIR/external-sources.md"

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

require_pattern 'docs/studies/README\.md' "skill routes through study records docs"
require_pattern 'docs/studies/snapback_shortening_effort/status\.md' "skill references pinned study status"
require_pattern 'docs/studies/snapback_shortening_effort/routes\.md' "skill references pinned study routes"
require_pattern 'docs/studies/snapback_shortening_effort/pipeline\.yaml' "skill references pinned study pipeline"
require_pattern 'cruncher\.data-plane\.cruncher-study-status --study-dir docs/studies/snapback_shortening_effort --json' "skill includes pinned snapshot command"
require_pattern 'cruncher\.data-plane\.cruncher-study-preflight --study-dir docs/studies/snapback_shortening_effort --scope next --json' "skill includes pinned preflight command"
require_pattern 'released-product Snapback' "skill preserves released-product boundary"
require_pattern 'YIU' "skill preserves YIU boundary language"
require_pattern 'harness-engineering' "skill routes harness work outward"
require_pattern 'pragmatic-programming-principles' "skill routes contract work outward"
require_pattern 'knowledge-integrity' "skill names knowledge-integrity endpoint"
require_pattern 'autonomy-capability' "skill names autonomy-capability endpoint"
require_pattern 'architecture-invariants' "skill names architecture-invariants endpoint"
require_pattern '\.agents/skills/snapback-hairpin-study/SKILL\.md' "root and cruncher AGENTS mention the skill" "$ROOT_AGENTS"
require_pattern '\.agents/skills/snapback-hairpin-study/SKILL\.md' "cruncher AGENTS mentions the skill" "$CRUNCHER_AGENTS"
require_pattern 'Pair with `harness-engineering`' "skill explains harness pairing" "$SKILL_FILE"
require_pattern 'Pair with `pragmatic-programming-principles`' "skill explains pragmatic pairing" "$SKILL_FILE"

if [[ $failures -eq 0 ]]; then
  printf 'Audit finished with no failures.\n'
else
  printf 'Audit finished with %d failure(s).\n' "$failures"
  exit 1
fi
