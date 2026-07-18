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
require_file "$SKILL_DIR/references/backend-scope.md"
require_file "$SKILL_DIR/references/chimerax-rest-contract.md"
require_file "$SKILL_DIR/references/live-session-contract.md"
require_file "$SKILL_DIR/references/command-allowlist.md"
require_file "$SKILL_DIR/references/natural-language-control-map.md"
require_file "$SKILL_DIR/references/pose-manifest-contract.md"
require_file "$SKILL_DIR/references/style-preset-contract.md"
require_file "$SKILL_DIR/references/molecular-scene-contract.md"
require_file "$SKILL_DIR/references/render-verification-contract.md"
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
require_file "$SCRIPT_DIR/chimerax-apply-complex-style.py"
require_file "$SCRIPT_DIR/chimerax-verify-render.py"

if [[ "$(basename "$SKILL_DIR")" == "chimerax-structure-review" ]]; then
  pass "skill folder names the ChimeraX-specific backend"
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

if grep -q -- "--version" "$SCRIPT_DIR/chimerax-preflight.sh"; then
  fail "preflight must not launch ChimeraX for version checks"
else
  pass "preflight avoids executable version launch"
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

if grep -q 'save_movie_frame' "$SKILL_DIR/references/command-allowlist.md" \
  && grep -q 'all four corners' "$SKILL_DIR/references/molecular-scene-contract.md"; then
  pass "fixed-size frame capture and background validation are documented"
else
  fail "fixed-size frame capture contract is incomplete"
fi

if grep -q 'Default ladder display' "$SKILL_DIR/references/external-sources.md" \
  && grep -q 'system command-line options' "$SKILL_DIR/references/external-sources.md"; then
  pass "ChimeraX nucleotide and startup claims cite official sources"
else
  fail "ChimeraX source provenance is incomplete"
fi

if grep -q 'open -n -a "$CHIMERAX_APP" --stdin /dev/null --stdout "$CHIMERAX_LOG" --stderr "$CHIMERAX_LOG"' "$SCRIPT_DIR/chimerax-session-start.sh" \
  && grep -q 'nohup "$CHIMERAX_BIN_RESOLVED" --script "$START_SCRIPT" </dev/null' "$SCRIPT_DIR/chimerax-session-start.sh" \
  && grep -q 'lsof -tiTCP:"$PORT" -sTCP:LISTEN' "$SCRIPT_DIR/chimerax-session-start.sh"; then
  pass "graphical session launcher uses detached platform paths and resolves the REST owner"
else
  fail "graphical session launcher must survive the invoking shell and record the REST owner"
fi

if grep -q 'uv run python .agents/skills/chimerax-structure-review/scripts/chimerax-apply-complex-style.py' \
  "$SKILL_DIR/references/molecular-scene-contract.md"; then
  pass "role-aware style examples use the repository Python runtime"
else
  fail "role-aware style examples must use the repository Python runtime"
fi

for script in "$SCRIPT_DIR"/chimerax-*.sh "$SCRIPT_DIR"/*.py "$SCRIPT_DIR"/audit-chimerax-structure-review-skill.sh; do
  if [[ -x "$script" ]]; then
    pass "script executable: ${script#$REPO_ROOT/}"
  else
    fail "script not executable: ${script#$REPO_ROOT/}"
  fi
done

if python3 - "$SCRIPT_DIR/chimerax-send-command.py" "$SCRIPT_DIR/chimerax-capture-pose.py" "$SCRIPT_DIR/chimerax-apply-complex-style.py" "$SCRIPT_DIR/chimerax-verify-render.py" <<'PY'
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

if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
  tmp_render_dir="$(mktemp -d)"
  ffmpeg -v error -f lavfi -i 'color=c=white:s=64x64:d=1' \
    -vf 'drawbox=x=16:y=16:w=32:h=32:color=black:t=fill' -frames:v 1 "$tmp_render_dir/valid.png"
  ffmpeg -v error -f lavfi -i 'color=c=black:s=64x64:d=1' \
    -vf 'drawbox=x=16:y=16:w=32:h=32:color=white:t=fill' -frames:v 1 "$tmp_render_dir/invalid.png"
  if "$SCRIPT_DIR/chimerax-verify-render.py" \
    --image "$tmp_render_dir/valid.png" \
    --expected-width 64 \
    --expected-height 64 \
    --background '#FFFFFF' \
    --minimum-content-extent 0.25 >/dev/null; then
    pass "render verifier accepts a dimensioned, framed still"
  else
    fail "render verifier rejected a valid still"
  fi
  if "$SCRIPT_DIR/chimerax-verify-render.py" \
    --image "$tmp_render_dir/invalid.png" \
    --expected-width 64 \
    --expected-height 64 \
    --background '#FFFFFF' >/dev/null 2>&1; then
    fail "render verifier accepted wrong-background corners"
  else
    pass "render verifier rejects wrong-background corners"
  fi
  rm -rf "$tmp_render_dir"
else
  fail "ffmpeg and ffprobe are required for render verification"
fi

tmp_open_dir="$(mktemp -d)"
touch "$tmp_open_dir/model.pdb" "$tmp_open_dir/unsafe.cxc"
if python3 - "$SCRIPT_DIR/chimerax-send-command.py" "$SCRIPT_DIR/chimerax-capture-pose.py" "$tmp_open_dir" <<'PY'
import datetime
import importlib.util
from pathlib import Path
import sys

send_path = Path(sys.argv[1])
capture_path = Path(sys.argv[2])
tmp_open_dir = Path(sys.argv[3])
if not hasattr(datetime, "UTC"):
    datetime.UTC = datetime.timezone.utc

send_spec = importlib.util.spec_from_file_location("chimerax_send_command", send_path)
send_module = importlib.util.module_from_spec(send_spec)
assert send_spec.loader is not None
send_spec.loader.exec_module(send_module)

capture_spec = importlib.util.spec_from_file_location("chimerax_capture_pose", capture_path)
capture_module = importlib.util.module_from_spec(capture_spec)
assert capture_spec.loader is not None
capture_spec.loader.exec_module(capture_module)

assert send_module._allowed(f"open {tmp_open_dir / 'model.pdb'}")
assert not send_module._allowed(f"open {tmp_open_dir / 'unsafe.cxc'}")
assert send_module._allowed("nucleotides #1/D,E,F atoms")
assert send_module._allowed("nucleotides #1/D,E,F ladder")
assert not send_module._allowed("nucleotides #1/D,E,F slab")
assert send_module._allowed("cartoon style nucleic xsect rectangle width 1.8 thick 0.25")
assert not send_module._allowed("cartoon style nucleic xsect triangle width 1.8 thick 0.25")
assert send_module._allowed("cartoon #1/D,E,F suppressBackboneDisplay true")
assert send_module._allowed("cartoon tether nucleic shape cylinder sides 8 scale 0.65 opacity 1")
assert send_module._allowed("label delete")
assert send_module._allowed("hide #1 pseudobonds")
assert send_module._allowed("size #1/D,E,F stickRadius 0.20")
assert send_module._allowed("name dna_role #1/D")
assert send_module._allowed("rename #1 molecular_complex")
assert send_module._allowed("color #1/A #E8E4DA target c")
assert send_module._allowed("color #1/D #B97700 target acf")
assert not send_module._allowed("color #1/D #B97700 target acfx")
assert not send_module._allowed("color #1/A #E8E4D target c")
assert not send_module._allowed("color #1/A #E8E4DAG target c")
assert send_module._allowed("shape ribbon #1/D@P width 1.4 height 0.12 followBonds false color gold modelId #2")
assert capture_module._cxc_quoted_path(tmp_open_dir / "pose.png", label="image") == f'"{tmp_open_dir / "pose.png"}"'
try:
    capture_module._cxc_quoted_path(tmp_open_dir / 'bad"name.png', label="image")
except ValueError:
    pass
else:
    raise AssertionError("quoted ChimeraX output path was accepted")
PY
then
  pass "ChimeraX helper allowlists reject executable open paths and unsafe CXC path text"
else
  fail "ChimeraX helper allowlists reject executable open paths and unsafe CXC path text"
fi
rm -rf "$tmp_open_dir"

if "$SCRIPT_DIR/chimerax-apply-complex-style.py" \
  --dry-run \
  --protein-selection '#1/A' \
  --dna-selection '#1/D' \
  --rna-selection '#1/E,F' \
  --nucleic-selection '#1/D,E,F' \
  | grep -q 'nucleotides #1/D,E,F ladder'; then
  pass "role-aware complex-style dry-run emits the default nucleotide ladder"
else
  fail "role-aware complex-style dry-run did not emit the default nucleotide ladder"
fi

if "$SCRIPT_DIR/chimerax-apply-complex-style.py" \
  --dry-run \
  --protein-selection '#1/A' \
  --dna-selection '#1/D' \
  --rna-selection '#1/E,F' \
  --nucleic-selection '#1/D,E,F' \
  | grep -q 'color #1/D #B97700 target acf' \
  && "$SCRIPT_DIR/chimerax-apply-complex-style.py" \
    --dry-run \
    --protein-selection '#1/A' \
    --dna-selection '#1/D' \
    --rna-selection '#1/E,F' \
    --nucleic-selection '#1/D,E,F' \
    | grep -q 'color #1/E,F #C84C5A target acf'; then
  pass "role-aware complex style uses stable, distinct DNA and RNA colors"
else
  fail "role-aware complex style lost stable DNA or RNA colors"
fi

if "$SCRIPT_DIR/chimerax-apply-complex-style.py" \
  --dry-run \
  --protein-selection '#1/A' \
  --dna-selection '#1/D' \
  --rna-selection '#1/E,F' \
  --nucleic-selection '#1/D,E,F' \
  | grep -q 'transparency #1/A 35 target s'; then
  pass "role-aware complex style uses 65 percent surface alpha"
else
  fail "role-aware complex style lost the 65 percent surface alpha contract"
fi

if "$SCRIPT_DIR/chimerax-apply-complex-style.py" \
  --dry-run \
  --protein-selection '#1/A' \
  --dna-selection '#1/D' \
  --rna-selection '#1/E,F' \
  --nucleic-selection '#1/D,E,F' \
  --nucleic-display connected-atoms \
  --nucleic-backbone-mode phosphate-ribbon \
  --dna-phosphate-selection '#1/D@P' \
  --rna-phosphate-selection '#1/E@P' \
  --rna-phosphate-selection '#1/F@P' \
  | grep -q 'rename #20 dna_backbone'; then
  pass "phosphate-ribbon fallback dry-run emits semantically named models"
else
  fail "phosphate-ribbon fallback dry-run did not emit semantically named models"
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

tmp_capture_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_capture_dir"' EXIT
for rejected_capture_args in \
  '--pose-id pose;exit --background-color white --title ok' \
  '--pose-id pose_ok --background-color "white; close session" --title ok' \
  '--pose-id pose_ok --background-color white --title "review; close session"'; do
  if bash -lc "\"$SCRIPT_DIR/chimerax-capture-pose.py\" --port 65535 --preopened-session --output-dir \"$tmp_capture_dir\" $rejected_capture_args" >/dev/null 2>&1; then
    fail "rejects unsafe capture options: $rejected_capture_args"
  else
    pass "rejects unsafe capture options: $rejected_capture_args"
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
