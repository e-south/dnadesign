#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../../.." && pwd)"
SKILL_FILE="$ROOT_DIR/SKILL.md"
REFERENCE_DIR="$ROOT_DIR/references"
ROOT_AGENTS="$REPO_ROOT/AGENTS.md"
USR_AGENTS="$REPO_ROOT/src/dnadesign/usr/AGENTS.md"
LEGACY_SKILL="$REPO_ROOT/src/dnadesign/usr/skills/bu-scc-usr-sync/SKILL.md"
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
  local path="${3:-$USR_AGENTS}"
  if grep -Fq -- "$needle" "$path"; then
    fail "$label"
  else
    pass "$label"
  fi
}

require_file "$SKILL_FILE"
require_file "$REFERENCE_DIR/sync-loop.md"
require_file "$REFERENCE_DIR/external-sources.md"
require_file "$ROOT_AGENTS"
require_file "$USR_AGENTS"
require_file "$REPO_ROOT/src/dnadesign/usr/docs/operations/sync/README.md"
require_file "$REPO_ROOT/src/dnadesign/usr/docs/operations/sync/setup.md"
require_file "$REPO_ROOT/docs/bu-scc/README.md"

for section in "## Scope" "## Success Criteria" "## Workflow" "## Required Deliverables" "## Output" "## Trigger Tests"; do
  require_text "$section" "section present: $section"
done

require_text ".agents/skills/bu-scc-usr-sync/SKILL.md" "root AGENTS routes to BU SCC sync skill" "$ROOT_AGENTS"
require_text ".agents/skills/bu-scc-usr-sync/SKILL.md" "usr AGENTS routes to BU SCC sync skill" "$USR_AGENTS"
require_absent "src/dnadesign/usr/skills/bu-scc-usr-sync/SKILL.md" "usr AGENTS avoids package-local BU SCC skill path"
require_text "src/dnadesign/usr/docs/operations/sync/README.md" "skill links sync route"
require_text "src/dnadesign/usr/docs/operations/sync/setup.md" "skill links sync setup doc"
require_text "docs/bu-scc/README.md" "skill links BU SCC docs"
require_text "--remotes-config <remotes.yaml>" "skill prefers explicit remotes config"
require_text "USR_REMOTES_PATH" "skill documents shell fallback"
require_text "uv run usr remotes doctor --remote <name>" "skill includes remotes doctor"
require_text "uv run usr remotes status --remote <name>" "skill includes remotes status"
require_text "uv run usr remotes warm-auth --remote <name>" "skill includes warm-auth path"
require_text "Never delete SCC datasets" "skill includes no-delete guardrail" "$REFERENCE_DIR/sync-loop.md"
require_text "records.parquet" "skill keeps real-dataset contract explicit"

if [[ -f "$LEGACY_SKILL" ]]; then
  fail "legacy package-local skill should be removed"
else
  pass "legacy package-local skill removed"
fi

if [[ $failures -eq 0 ]]; then
  printf 'Audit finished with no failures.\n'
else
  printf 'Audit finished with %d failure(s).\n' "$failures"
  exit 1
fi
