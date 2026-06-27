#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SKILL_DIR/../../.." && pwd)"
SKILL_FILE="$SKILL_DIR/SKILL.md"
failures=0

pass() { printf 'PASS: %s\n' "$1"; }
fail() { printf 'FAIL: %s\n' "$1"; failures=$((failures + 1)); }

require_file() {
  local path="$1"
  [[ -f "$path" ]] && pass "found ${path#$REPO_ROOT/}" || fail "missing ${path#$REPO_ROOT/}"
}

require_absent_in_skill() {
  local pattern="$1"
  local label="$2"
  if grep -Eiq "$pattern" "$SKILL_FILE"; then
    fail "$label"
  else
    pass "$label"
  fi
}

require_file "$SKILL_FILE"
require_file "$SKILL_DIR/agents/openai.yaml"
require_file "$SKILL_DIR/references/external-sources.md"
require_file "$SKILL_DIR/references/workflow-router.md"
require_file "$SKILL_DIR/references/first-run.md"
require_file "$SKILL_DIR/references/collaboration-cadence.md"
require_file "$SKILL_DIR/references/name-scope-decision.md"
require_file "$SKILL_DIR/references/chimerax-rest-contract.md"
require_file "$SKILL_DIR/references/live-session-contract.md"
require_file "$SKILL_DIR/references/command-allowlist.md"
require_file "$SKILL_DIR/references/natural-language-control-map.md"
require_file "$SKILL_DIR/references/pose-manifest-contract.md"
require_file "$SKILL_DIR/references/style-preset-contract.md"
require_file "$SKILL_DIR/references/sibling-patterns-example.md"
require_file "$SKILL_DIR/references/test-matrix.md"
require_file "$SKILL_DIR/assets/pose_manifest.schema.yaml"
require_file "$SKILL_DIR/assets/control_session_manifest.schema.yaml"
require_file "$SKILL_DIR/assets/live_session_manifest.schema.yaml"
require_file "$SKILL_DIR/assets/style_presets.yaml"
require_file "$SKILL_DIR/assets/demo_structure.pdb"
require_file "$SCRIPT_DIR/chimerax-preflight.sh"
require_file "$SCRIPT_DIR/chimerax-rest-smoke.sh"
require_file "$SCRIPT_DIR/chimerax-live-demo.sh"
require_file "$SCRIPT_DIR/chimerax-session-start.sh"
require_file "$SCRIPT_DIR/chimerax-session-status.sh"
require_file "$SCRIPT_DIR/chimerax-session-stop.sh"
require_file "$SCRIPT_DIR/chimerax-send-command.py"
require_file "$SCRIPT_DIR/chimerax-capture-pose.py"

if [[ "$(basename "$SKILL_DIR")" == "chimerax-structure-review" ]]; then
  pass "skill folder uses generic structure-review name"
else
  fail "skill folder name is not chimerax-structure-review"
fi

if grep -q "name: chimerax-structure-review" "$SKILL_FILE"; then
  pass "frontmatter name matches folder"
else
  fail "frontmatter name mismatch"
fi

for heading in "## Scope" "## Required Inputs" "## Success Criteria" "## Workflow" "## Guardrails" "## Required Deliverables" "## Trigger Tests"; do
  if grep -Fxq "$heading" "$SKILL_FILE"; then
    pass "heading present: $heading"
  else
    fail "heading missing: $heading"
  fi
done

require_absent_in_skill 'Eco1|Ec86|retron|reverse transcriptase|strand displacement|hairpin' "top-level skill avoids study-specific biology"

if grep -q "Retrieved: " "$SKILL_DIR/references/external-sources.md"; then
  pass "external sources include retrieved date"
else
  fail "external sources missing retrieved date"
fi

if grep -q "remotecontrol rest start" "$SKILL_DIR/references/chimerax-rest-contract.md" && grep -q "127.0.0.1" "$SKILL_DIR/references/chimerax-rest-contract.md"; then
  pass "REST contract names local remotecontrol path"
else
  fail "REST contract missing local remotecontrol path"
fi

if grep -q "session-ready" "$SKILL_DIR/references/collaboration-cadence.md" && grep -q "stop-or-continue" "$SKILL_DIR/references/collaboration-cadence.md"; then
  pass "collaboration pause points documented"
else
  fail "collaboration pause points missing"
fi

if grep -q "schema_id: chimerax_control_session_v1" "$SKILL_DIR/assets/control_session_manifest.schema.yaml" \
  && grep -q "command_log_path" "$SKILL_DIR/assets/control_session_manifest.schema.yaml"; then
  pass "control-session schema captures live collaboration handle"
else
  fail "control-session schema missing required collaboration fields"
fi

if grep -Eiq 'arbitrary|raw user prose|free-text' "$SKILL_FILE" "$SKILL_DIR/references/command-allowlist.md"; then
  pass "arbitrary command guardrails documented"
else
  fail "arbitrary command guardrails missing"
fi

if grep -Eq 'runscript|shell|python ' "$SKILL_DIR/references/command-allowlist.md"; then
  pass "forbidden high-risk command patterns documented"
else
  fail "forbidden command patterns missing"
fi

if grep -Eq '^\| Scenario \| Prompt \| Expected Behavior \| Pass/Fail \|$' "$SKILL_DIR/references/test-matrix.md"; then
  pass "test matrix uses scenario table"
else
  fail "test matrix missing scenario table"
fi

for script in "$SCRIPT_DIR"/chimerax-*.sh "$SCRIPT_DIR"/*.py "$SCRIPT_DIR"/audit-chimerax-structure-review-skill.sh; do
  if [[ -x "$script" ]]; then
    pass "script executable: ${script#$REPO_ROOT/}"
  else
    fail "script not executable: ${script#$REPO_ROOT/}"
  fi
done

if python3 - "$SCRIPT_DIR/chimerax-send-command.py" "$SCRIPT_DIR/chimerax-capture-pose.py" <<'PY'
import ast
from pathlib import Path
import sys

for raw_path in sys.argv[1:]:
    ast.parse(Path(raw_path).read_text(encoding="utf-8"), filename=raw_path)
PY
then
  pass "python helper scripts parse"
else
  fail "python helper scripts parse"
fi

for rejected_command in \
  'open https://example.org/1abc.pdb' \
  'remotecontrol rest stop; exit' \
  'save "/tmp/not_allowed.txt"' \
  'shell rm -rf /'; do
  if "$SCRIPT_DIR/chimerax-send-command.py" --port 65535 --command "$rejected_command" >/dev/null 2>&1; then
    fail "rejects unsafe command: $rejected_command"
  else
    pass "rejects unsafe command: $rejected_command"
  fi
done

for shell_script in "$SCRIPT_DIR"/chimerax-*.sh "$SCRIPT_DIR"/audit-chimerax-structure-review-skill.sh; do
  if bash -n "$shell_script"; then
    pass "shell script parses: ${shell_script#$REPO_ROOT/}"
  else
    fail "shell script parses: ${shell_script#$REPO_ROOT/}"
  fi
done

if [[ -e "$REPO_ROOT/outputs" ]]; then
  fail "repo-root outputs/ exists"
else
  pass "repo-root outputs/ absent"
fi

if (( failures > 0 )); then
  printf 'Audit finished with %d failure(s).\n' "$failures"
  exit 1
fi
printf 'Audit finished with 0 failures.\n'
