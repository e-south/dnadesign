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

require_frontmatter_yaml() {
  if uv run python - "$SKILL_FILE" <<'PY'
from pathlib import Path
import sys

import yaml

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
if not text.startswith("---\n"):
    raise SystemExit("missing opening frontmatter delimiter")
try:
    frontmatter = text.split("---", 2)[1]
except IndexError as exc:
    raise SystemExit("missing closing frontmatter delimiter") from exc
payload = yaml.safe_load(frontmatter)
if not isinstance(payload, dict):
    raise SystemExit("frontmatter is not a mapping")
if payload.get("name") != path.parent.name:
    raise SystemExit("frontmatter name does not match skill folder")
description = payload.get("description")
if not isinstance(description, str) or not description.strip():
    raise SystemExit("frontmatter description is missing")
if len(description) > 260:
    raise SystemExit(f"frontmatter description exceeds hard budget: {len(description)} > 260")
if "Snapback/scar-nick/YIU" in description:
    raise SystemExit("frontmatter should not positively enumerate primitive route names")
if "generic Cruncher/snapback" not in description:
    raise SystemExit("frontmatter must route generic Cruncher/snapback requests away")
metadata = payload.get("metadata")
if not isinstance(metadata, dict):
    raise SystemExit("frontmatter metadata is missing")
version = metadata.get("version")
if not isinstance(version, str) or not version.count(".") == 2:
    raise SystemExit("frontmatter metadata.version must be semver-shaped")
PY
  then
    pass "frontmatter parses as YAML and stays within discovery budget"
  else
    fail "frontmatter parses as YAML and stays within discovery budget"
  fi
}

require_file "$SKILL_FILE"
require_file "$ROOT_AGENTS"
require_file "$CRUNCHER_AGENTS"
require_file "$REPO_ROOT/docs/studies/README.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/status.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/routes.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/scar-nick-base-junction.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/linear-ssdna-composition.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/msd_design_registry.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/msd_design_hit_labels.txt"
require_file "$REPO_ROOT/docs/dev/plans/2026-05-13-generic-linear-ssdna-composition-spec.md"
require_file "$REPO_ROOT/docs/exec-plans/completed/2026-05-13-generic-linear-ssdna-composition.md"
require_file "$REPO_ROOT/docs/exec-plans/active/2026-05-14-linear-ssdna-composition-hardening-followups.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/pipeline.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/ops.study.yaml"
require_file "$REPO_ROOT/src/dnadesign/cruncher/docs/operations/cruncher-study-status.md"
require_file "$REPO_ROOT/src/dnadesign/cruncher/docs/operations/cruncher-study-preflight.md"
require_file "$REFERENCE_DIR/route-matrix.md"
require_file "$REFERENCE_DIR/refresh-loop.md"
require_file "$REFERENCE_DIR/study-surfaces.md"
require_file "$REFERENCE_DIR/msd-design-references.md"
require_file "$REFERENCE_DIR/test-matrix.md"
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
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/linear-ssdna-composition.md"
  printf '\n'
  cat "$REPO_ROOT/docs/dev/plans/2026-05-13-generic-linear-ssdna-composition-spec.md"
  printf '\n'
  cat "$REPO_ROOT/docs/exec-plans/completed/2026-05-13-generic-linear-ssdna-composition.md"
  printf '\n'
  cat "$REPO_ROOT/docs/exec-plans/active/2026-05-14-linear-ssdna-composition-hardening-followups.md"
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
require_frontmatter_yaml

require_pattern 'docs/studies/README\.md' "skill routes through study records docs"
require_pattern 'docs/studies/retron_hairpin_design/status\.md' "skill references pinned study status"
require_pattern 'docs/studies/retron_hairpin_design/routes\.md' "skill references pinned study routes"
require_pattern 'docs/studies/retron_hairpin_design/scar-nick-base-junction\.md' "skill references scar-nick base-junction context"
require_pattern 'docs/studies/retron_hairpin_design/linear-ssdna-composition\.md' "skill references linear ssDNA composition handoff"
require_pattern 'docs/studies/retron_hairpin_design/msd_design_registry\.yaml' "skill references MSD design registry"
require_pattern 'docs/studies/retron_hairpin_design/msd_design_hit_labels\.txt' "skill references MSD selected labels"
require_pattern 'docs/dev/plans/2026-05-13-generic-linear-ssdna-composition-spec\.md' "skill references linear ssDNA dev spec"
require_pattern 'docs/exec-plans/completed/2026-05-13-generic-linear-ssdna-composition\.md' "skill references linear ssDNA implementation record"
require_pattern 'docs/exec-plans/active/2026-05-14-linear-ssdna-composition-hardening-followups\.md' "skill references linear ssDNA follow-up plan"
require_pattern 'docs/studies/retron_hairpin_design/pipeline\.yaml' "skill references pinned study pipeline"
require_pattern 'references/test-matrix\.md' "skill references test matrix for validation"
require_pattern 'developers\.openai\.com/api/docs/guides/prompt-engineering#coding' "skill records OpenAI Developers prompt-surface source"
require_pattern 'Start with input completeness, not study phase' "skill uses compiler-first routing"
require_pattern 'status command output' "skill keeps status as optional progress evidence"
require_pattern 'released-product Snapback' "skill preserves released-product boundary"
require_pattern 'scar-nick' "skill preserves scar-nick boundary language"
require_pattern 'S0=M' "skill preserves scar-compatible ligation invariant"
require_pattern 'generic linear ssDNA composition' "skill preserves composition route language"
require_pattern 'msd_design_reference_v1' "skill preserves MSD design-reference contract"
require_pattern 'msd_design_catalog_v1' "skill preserves MSD design-catalog contract"
require_pattern 'top-level `retron-msd`' "skill rejects top-level retron-msd leakage"
require_pattern 'Do not say "snapshot posture"' "skill rejects snapshot-posture output"
require_pattern 'four-base left/right basal spans' "skill preserves scar-nick projection boundary"
require_pattern 'YIU' "skill preserves YIU boundary language"
require_pattern 'harness-engineering' "skill routes harness work outward"
require_pattern 'code-change-discipline' "skill routes contract work outward"
require_pattern '\.agents/skills/retron-hairpin-study/SKILL\.md' "root AGENTS mention the skill" "$ROOT_AGENTS"
require_pattern '\.agents/skills/retron-hairpin-study/SKILL\.md' "cruncher AGENTS mentions the skill" "$CRUNCHER_AGENTS"
reject_pattern "$OLD_SKILL_NAME" "old skill route removed"
reject_pattern 'whether the answer came from snapshot posture' "top-level skill no longer requires snapshot posture" "$SKILL_FILE"
reject_pattern 'current phase and next route' "top-level skill no longer reports phase by default" "$SKILL_FILE"
require_pattern 'Pair with `harness-engineering`' "skill explains harness pairing" "$SKILL_FILE"
require_pattern 'Pair with `code-change-discipline`' "skill explains code-change pairing" "$SKILL_FILE"
require_pattern 'Fresh/naive agent' "test matrix covers naive-agent discovery"
require_pattern 'Finder reveal command' "test matrix covers GenBank Finder handoff"

if [[ $failures -eq 0 ]]; then
  printf 'Audit finished with no failures.\n'
else
  printf 'Audit finished with %d failure(s).\n' "$failures"
  exit 1
fi
