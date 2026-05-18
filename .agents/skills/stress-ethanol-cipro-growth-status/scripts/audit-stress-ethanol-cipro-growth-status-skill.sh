#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../../.." && pwd)"
SKILL_FILE="$ROOT_DIR/SKILL.md"
REFERENCE_DIR="$ROOT_DIR/references"
failures=0

pass() { printf 'PASS: %s\n' "$1"; }
fail() { printf 'FAIL: %s\n' "$1"; failures=$((failures + 1)); }

require_file() {
  local path="$1"
  [[ -f "$path" ]] && pass "found ${path#$REPO_ROOT/}" || fail "missing file $path"
}

require_section() {
  local section="$1"
  grep -Fxq "$section" "$SKILL_FILE" && pass "section present: $section" || fail "section missing: $section"
}

require_pattern() {
  local pattern="$1"
  local label="$2"
  local path="${3:-$SKILL_FILE}"
  grep -Eq "$pattern" "$path" && pass "$label" || fail "$label"
}

require_absent() {
  local pattern="$1"
  local label="$2"
  local path="${3:-$SKILL_FILE}"
  if grep -Eq "$pattern" "$path"; then fail "$label"; else pass "$label"; fi
}

require_max_lines() {
  local path="$1"
  local max_lines="$2"
  local label="$3"
  local line_count
  line_count="$(wc -l < "$path" | tr -d ' ')"
  if [[ "$line_count" -le "$max_lines" ]]; then
    pass "$label (${line_count}/${max_lines})"
  else
    fail "$label (${line_count}/${max_lines})"
  fi
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
if "Do not use for another study or for family-level routing" not in description:
    raise SystemExit("frontmatter must reject other studies and family-level routing")
metadata = payload.get("metadata")
if not isinstance(metadata, dict):
    raise SystemExit("frontmatter metadata is missing")
version = metadata.get("version")
if not isinstance(version, str) or version.count(".") != 2:
    raise SystemExit("frontmatter metadata.version must be semver-shaped")
PY
  then
    pass "frontmatter parses as YAML and stays within discovery budget"
  else
    fail "frontmatter parses as YAML and stays within discovery budget"
  fi
}

require_file "$SKILL_FILE"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/catalog/contracts/status.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/catalog/contracts/preflight.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/ops.study.yaml"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/contract/surfaces/execution/commands/notify/profile-doctor.yaml"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/contract/surfaces/execution/commands/notify/resolve-events.yaml"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/contract/readiness/checks/infer_batch_preparation/sequence-views.yaml"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/contract/readiness/checks/infer_batch_preparation/completion.yaml"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/contract/readiness/checks/infer_batch_preparation/notify.yaml"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/README.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/source/densegen.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/compute/infer.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/source/construct.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/analysis/cluster.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/decision/opal/README.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/decision/opal/campaign-commands.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/analysis/latentdna.md"
require_file "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/contexts/opal/candidate-table.md"
require_file "$REPO_ROOT/src/dnadesign/studies/studies/stress_ethanol_cipro_growth/status/ops/status.registry.yaml"
require_file "$REPO_ROOT/src/dnadesign/studies/studies/stress_ethanol_cipro_growth/status/service.py"
require_file "$REPO_ROOT/src/dnadesign/studies/studies/stress_ethanol_cipro_growth/status/snapshot.py"
require_file "$REPO_ROOT/src/dnadesign/studies/studies/stress_ethanol_cipro_growth/status/preflight.py"
require_file "$REPO_ROOT/src/dnadesign/studies/studies/stress_ethanol_cipro_growth/status/probes/runtime_dependencies.py"
require_file "$REPO_ROOT/src/dnadesign/studies/studies/stress_ethanol_cipro_growth/status/probes/semantic_completeness.py"
require_file "$REPO_ROOT/src/dnadesign/studies/studies/stress_ethanol_cipro_growth/status/probes/sequence_view_contracts.py"
require_file "$REFERENCE_DIR/route-matrix.md"
require_file "$REFERENCE_DIR/refresh-loop.md"
require_file "$REFERENCE_DIR/study-surfaces.md"
require_file "$REFERENCE_DIR/external-sources.md"

require_frontmatter_yaml
require_section "## Required Deliverables"
require_section "## Trigger Tests"
require_pattern '^name: stress-ethanol-cipro-growth-status$' "skill name is study-specific"
require_pattern 'studies\.stress-ethanol-cipro-growth\.status' "skill names study status command"
require_pattern 'studies\.stress-ethanol-cipro-growth\.preflight' "skill names study preflight command"
require_pattern 'stress_ethanol_cipro_growth' "skill anchors the concrete study id"
require_pattern 'Do not use for another study or for family-level routing' "skill rejects family-level routing"
require_pattern 'Do not generalize it to another study' "skill guardrail rejects cross-study reuse"
require_pattern 'OPAL candidate-table details are meaningful only in this study' "skill keeps OPAL table study-owned"
require_pattern 'status/probes/' "skill reference exposes probe subpackage" "$REFERENCE_DIR/study-surfaces.md"
require_pattern '^## References$' "skill exposes progressive-disclosure references"
require_pattern 'routes/decision/opal/README\.md' "skill routes OPAL detail after one-hop map"
require_pattern 'routes/analysis/latentdna\.md' "skill routes LatentDNA detail after one-hop map"

require_absent 'promoter-study-status' "skill has no old status kind"
require_absent 'promoter-study-preflight' "skill has no old preflight kind"
require_absent 'usr\.data-plane\.promoter-study' "skill has no old registry id"
require_absent 'status_adapters/promoter_status' "skill has no old adapter path"
require_absent 'generic promoter' "skill avoids generic promoter routing language"

if [[ -e "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/contract/surfaces/execution/commands/notify.yaml" ]]; then
  fail "stress Notify commands are split by subcommand family"
else
  pass "stress Notify commands are split by subcommand family"
fi

if [[ -e "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/contract/readiness/checks/infer_batch_preparation.yaml" ]]; then
  fail "stress Infer readiness checks are split by owner/action lane"
else
  pass "stress Infer readiness checks are split by owner/action lane"
fi

require_max_lines "$REPO_ROOT/src/dnadesign/studies/studies/stress_ethanol_cipro_growth/status/service.py" 320 "status service stays orchestration-sized"
require_max_lines "$REPO_ROOT/src/dnadesign/studies/studies/stress_ethanol_cipro_growth/status/probes/runtime_dependencies.py" 140 "runtime probe module stays bounded"
require_max_lines "$REPO_ROOT/src/dnadesign/studies/studies/stress_ethanol_cipro_growth/status/probes/semantic_completeness.py" 200 "semantic-completeness probe module stays bounded"
require_max_lines "$REPO_ROOT/src/dnadesign/studies/studies/stress_ethanol_cipro_growth/status/probes/sequence_view_contracts.py" 240 "sequence-view probe module stays bounded"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/README.md" 140 "stress study route map stays one-hop"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/source/densegen.md" 80 "DenseGen route detail stays bounded"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/compute/infer.md" 80 "Infer route detail stays bounded"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/source/construct.md" 80 "Construct route detail stays bounded"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/analysis/cluster.md" 80 "Cluster route detail stays bounded"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/decision/opal/README.md" 100 "OPAL route detail stays bounded"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/decision/opal/campaign-commands.md" 80 "OPAL command detail stays bounded"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/routes/analysis/latentdna.md" 120 "LatentDNA route detail stays bounded"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/contract/readiness/checks/infer_batch_preparation/sequence-views.yaml" 120 "Infer sequence-view readiness fragment stays bounded"
require_max_lines "$REPO_ROOT/docs/studies/stress_ethanol_cipro_growth/operations/contract/readiness/checks/infer_batch_preparation/notify.yaml" 100 "Infer Notify readiness fragment stays bounded"

if [[ $failures -eq 0 ]]; then
  printf 'Audit finished with no failures.\n'
else
  printf 'Audit finished with %d failure(s).\n' "$failures"
  exit 1
fi
