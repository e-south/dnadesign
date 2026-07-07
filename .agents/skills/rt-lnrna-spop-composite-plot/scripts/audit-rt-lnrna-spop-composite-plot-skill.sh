#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../../.." && pwd)"
SKILL_FILE="$ROOT_DIR/SKILL.md"
failures=0

pass() { printf 'PASS: %s\n' "$1"; }
fail() { printf 'FAIL: %s\n' "$1"; failures=$((failures + 1)); }

require_file() {
  local path="$1"
  [[ -f "$path" ]] && pass "found $path" || fail "missing file $path"
}

require_pattern() {
  local pattern="$1"
  local path="$2"
  local label="$3"
  grep -Eq "$pattern" "$path" && pass "$label" || fail "$label"
}

require_file "$SKILL_FILE"
require_file "$ROOT_DIR/references/external-sources.md"
require_file "$ROOT_DIR/references/test-matrix.md"
require_file "$REPO_ROOT/docs/studies/rt_lnrna_sponging_construct_triage/routes/reader-spop-condition-structure-matrix.md"
require_file "$REPO_ROOT/src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_composite/materialize.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_composite/conditions.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_composite/condition_matrix.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_composite/tables.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_composite/structure_manifest.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_composite/render.py"

line_count="$(wc -l < "$SKILL_FILE" | tr -d ' ')"
if (( line_count <= 90 )); then
  pass "SKILL.md line budget respected"
else
  fail "SKILL.md exceeds 90 lines"
fi

if uv run python - "$SKILL_FILE" <<'PY'
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
payload = yaml.safe_load(text.split("---", 2)[1])
if payload["name"] != path.parent.name:
    raise SystemExit("frontmatter name mismatch")
if len(payload["description"]) > 220:
    raise SystemExit("description too long")
if "version" not in payload.get("metadata", {}):
    raise SystemExit("missing metadata.version")
if "Do not use" not in payload["description"]:
    raise SystemExit("description missing negative scope")
PY
then
  pass "frontmatter parses and stays within budget"
else
  fail "frontmatter parses and stays within budget"
fi

require_pattern "SPOP heatmap" "$SKILL_FILE" "positive trigger language present"
require_pattern "condition-structure (matrix|matrices)" "$SKILL_FILE" "route language present"
require_pattern "masked gray, not zero" "$SKILL_FILE" "missing-cell policy present"
require_pattern "normalized derepression" "$SKILL_FILE" "normalization policy present"
require_pattern "square" "$SKILL_FILE" "square heatmap tile policy present"
require_pattern "darker seagreen" "$SKILL_FILE" "palette policy present"
require_pattern "margins trimmed" "$SKILL_FILE" "thumbnail crop policy present"
require_pattern "retron-hairpin only when" "$SKILL_FILE" "hairpin route gate present"
require_pattern "generic LatentDNA" "$SKILL_FILE" "generic LatentDNA route-away present"
require_pattern "reader_spop_composite\\.materialize" "$SKILL_FILE" "materializer command path present"
require_pattern "Checked" "$ROOT_DIR/references/external-sources.md" "external source freshness column present"

if (( failures > 0 )); then
  printf 'Skill audit failed with %s failure(s).\n' "$failures" >&2
  exit 1
fi
printf 'Skill audit passed.\n'
