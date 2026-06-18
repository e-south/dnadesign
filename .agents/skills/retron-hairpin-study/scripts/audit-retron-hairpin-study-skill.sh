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
EXPECTED_REFERENCE_FILES=(
  external-sources.md
  msd-design-references.md
  refresh-loop.md
  route-matrix.md
  study-surfaces.md
  test-matrix.md
)

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
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/record/status.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/record/evidence/design-evidence.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/routes/README.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/compiler/README.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/contexts/README.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/routes/compiler/msd-design-references.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/routes/product/released-product-snapback.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/routes/product/scar-nick-base-junction.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/routes/composition/linear-ssdna-composition.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/routes/quality/yiu-boundary-check.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/contexts/cruncher/scar-nick-base-junction.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/contexts/composition/linear-ssdna-composition.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/compiler/catalog/msd_design_registry.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/compiler/catalog/msd_cap_sources.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/compiler/inputs/msd_design_hit_labels.txt"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/compiler/inputs/msd_design_177_194_cap_sources_spec.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/README.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/ontology/README.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/ontology/directions.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/design_sets/README.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/design_sets/scar_nick_profile_panel_v1.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/provenance/README.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/provenance/compiler_runs/README.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/provenance/compiler_runs/2026-05-18-msd-177-194.compile.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/provenance/materializations/README.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/provenance/materializations/2026-05-18-msd-177-194.single-unit.yaml"
require_file "$REPO_ROOT/docs/dev/plans/cross-tool/linear-ssdna-composition/2026-05-13-generic-linear-ssdna-composition.md"
require_file "$REPO_ROOT/docs/exec-plans/completed/2026-05-13-generic-linear-ssdna-composition.md"
require_file "$REPO_ROOT/docs/exec-plans/active/2026-05-14-linear-ssdna-composition-hardening-followups.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/runtime/command-groups/pipeline.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/runtime/command-groups/README.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/runtime/command-groups/lanes/compiler.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/runtime/command-groups/lanes/materialize.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/runtime/command-groups/lanes/snapback.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/runtime/command-groups/lanes/scar-nick.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/runtime/command-groups/lanes/yiu.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/ops.study.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/contract/surfaces/execution/workspaces.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/contract/surfaces/execution/commands/compiler.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/contract/surfaces/execution/commands/materialize.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/contract/surfaces/execution/commands/snapback.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/contract/surfaces/execution/commands/yiu.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/contract/readiness/scope.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/contract/readiness/group-bindings.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/contract/readiness/next-scope.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/contract/readiness/checks/context_consolidation.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/contract/readiness/checks/msd_design_reference_catalog.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/contract/readiness/checks/msd_single_unit_materialize.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/catalog/contracts/status.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/catalog/contracts/preflight.md"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/compiler/references.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/compiler/catalog_bundle.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/compiler/materialization.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/compiler/exceptions.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/interfaces/cli/app.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/interfaces/cli/inputs.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/interfaces/cli/io.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/interfaces/cli/messages.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/catalog/compiler_spec.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/catalog/msd_ids.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/catalog/registry.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/catalog/sequence_inputs.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/artifact_contracts/composition_payload.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/artifact_contracts/output_guards.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/artifact_contracts/materialized_outputs.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/artifact_contracts/manifests.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/artifact_contracts/layout.py"
require_file "$REFERENCE_DIR/route-matrix.md"
require_file "$REFERENCE_DIR/refresh-loop.md"
require_file "$REFERENCE_DIR/study-surfaces.md"
require_file "$REFERENCE_DIR/msd-design-references.md"
require_file "$REFERENCE_DIR/test-matrix.md"
require_file "$REFERENCE_DIR/external-sources.md"

for ref in "$REFERENCE_DIR"/*.md; do
  ref_name="$(basename "$ref")"
  listed=false
  for expected in "${EXPECTED_REFERENCE_FILES[@]}"; do
    if [[ "$ref_name" == "$expected" ]]; then
      listed=true
      break
    fi
  done
  if [[ "$listed" == true ]]; then
    pass "reference file is listed: $ref_name"
  else
    fail "unlisted reference file: $ref_name"
  fi
done

for stale_surface in cli.py compiler.py errors.py; do
  if [[ -e "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/$stale_surface" ]]; then
    fail "root Retron study Python surface removed: $stale_surface"
  else
    pass "root Retron study Python surface removed: $stale_surface"
  fi
done

if [[ -e "$OLD_SKILL_DIR" ]]; then
  fail "old skill directory removed"
else
  pass "old skill directory removed"
fi

if [[ -e "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/contract/surfaces/execution_surfaces.yaml" ]]; then
  fail "Retron execution surfaces split into semantic fragments"
else
  pass "Retron execution surfaces split into semantic fragments"
fi

if [[ -e "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/contract/readiness/preflight.yaml" ]]; then
  fail "Retron readiness checks split into semantic fragments"
else
  pass "Retron readiness checks split into semantic fragments"
fi

cat "$SKILL_FILE" > "$TMP_COMBINED"
for ref_name in "${EXPECTED_REFERENCE_FILES[@]}"; do
  ref="$REFERENCE_DIR/$ref_name"
  printf '\n' >> "$TMP_COMBINED"
  cat "$ref" >> "$TMP_COMBINED"
done
{
  printf '\n'
  cat "$ROOT_AGENTS"
  printf '\n'
  cat "$CRUNCHER_AGENTS"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/README.md"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/record/status.md"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/record/evidence/design-evidence.md"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/routes/README.md"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/compiler/README.md"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/contexts/README.md"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/routes/compiler/msd-design-references.md"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/routes/product/released-product-snapback.md"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/routes/product/scar-nick-base-junction.md"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/routes/composition/linear-ssdna-composition.md"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/routes/quality/yiu-boundary-check.md"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/README.md"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/ontology/README.md"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/ontology/directions.yaml"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/design_sets/README.md"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/design_sets/scar_nick_profile_panel_v1.yaml"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/provenance/README.md"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/contexts/composition/linear-ssdna-composition.md"
  printf '\n'
  cat "$REPO_ROOT/docs/dev/plans/cross-tool/linear-ssdna-composition/2026-05-13-generic-linear-ssdna-composition.md"
  printf '\n'
  cat "$REPO_ROOT/docs/exec-plans/completed/2026-05-13-generic-linear-ssdna-composition.md"
  printf '\n'
  cat "$REPO_ROOT/docs/exec-plans/active/2026-05-14-linear-ssdna-composition-hardening-followups.md"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/runtime/command-groups/pipeline.yaml"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/runtime/command-groups/README.md"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/operations/ops.study.yaml"
} >> "$TMP_COMBINED"

require_section "## Scope"
require_section "## Success Criteria"
require_section "## Workflow"
require_section "## Required Deliverables"
require_section "## Output"
require_section "## Trigger Tests"
require_line_budget "$SKILL_FILE" "$SIZE_BUDGET_LINES"
require_frontmatter_yaml

if uv run python - "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design" <<'PY'
from pathlib import Path
import sys

source_root = Path(sys.argv[1])
budgets = {
    "compiler/references.py": 180,
    "compiler/catalog_bundle.py": 220,
    "compiler/materialization.py": 260,
    "compiler/exceptions.py": 60,
    "interfaces/cli/app.py": 360,
    "interfaces/cli/inputs.py": 140,
    "interfaces/cli/io.py": 140,
    "interfaces/cli/messages.py": 180,
    "artifact_contracts/composition_payload.py": 450,
    "artifact_contracts/output_guards.py": 450,
    "artifact_contracts/materialized_outputs.py": 450,
    "artifact_contracts/manifests.py": 450,
    "catalog/cap_sources.py": 220,
    "catalog/compiler_spec.py": 450,
    "catalog/compiler_spec_io.py": 140,
    "catalog/sequence_inputs.py": 120,
}

def implementation_line_count(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    if (
        len(lines) >= 10
        and lines[0] == '"""'
        and lines[1] == "-" * 80
        and lines[8] == "-" * 80
        and lines[9] == '"""'
    ):
        return len(lines) - 10
    return len(lines)

violations = []
for filename, budget in budgets.items():
    line_count = implementation_line_count(source_root / filename)
    if line_count > budget:
        violations.append(f"{filename} has {line_count} lines > {budget}")
if violations:
    raise SystemExit("; ".join(violations))
PY
then
  pass "Retron compiler source stays decomposed by responsibility"
else
  fail "Retron compiler source stays decomposed by responsibility"
fi

require_pattern 'docs/studies/README\.md' "skill routes through study records docs"
require_pattern 'docs/studies/retron_hairpin_design/record/status\.md' "skill references pinned study status"
require_pattern 'docs/studies/retron_hairpin_design/routes/README\.md' "skill references pinned study routes"
require_pattern 'references/msd-design-references\.md' "skill references MSD route detail"
require_pattern 'docs/studies/retron_hairpin_design/workbench/README\.md' "skill references workbench entrypoint"
require_pattern 'workbench/ontology' "skill references workbench ontology lane"
require_pattern 'workbench/provenance' "skill references workbench provenance lane"
require_pattern 'scar_nick_profile_panel_v1\.yaml' "skill references workbench design set"
require_pattern 'docs/studies/retron_hairpin_design/contexts/cruncher/scar-nick-base-junction\.md' "skill references scar-nick base-junction context"
require_pattern 'docs/studies/retron_hairpin_design/contexts/composition/linear-ssdna-composition\.md' "skill references linear ssDNA composition handoff"
require_pattern 'docs/studies/retron_hairpin_design/compiler/catalog/msd_design_registry\.yaml' "skill references MSD design registry"
require_pattern 'docs/studies/retron_hairpin_design/compiler/catalog/msd_cap_sources\.yaml' "skill references cap source lookup"
require_pattern 'docs/studies/retron_hairpin_design/compiler/inputs/msd_design_hit_labels\.txt' "skill references MSD selected labels"
require_pattern 'docs/studies/retron_hairpin_design/compiler/inputs/msd_design_177_194_cap_sources_spec\.yaml' "skill references full cohort materialization spec"
require_pattern 'docs/dev/plans/cross-tool/linear-ssdna-composition/2026-05-13-generic-linear-ssdna-composition.md' "skill references linear ssDNA dev spec"
require_pattern 'docs/exec-plans/completed/2026-05-13-generic-linear-ssdna-composition\.md' "skill references linear ssDNA implementation record"
require_pattern 'docs/exec-plans/active/2026-05-14-linear-ssdna-composition-hardening-followups\.md' "skill references linear ssDNA follow-up plan"
require_pattern 'docs/studies/retron_hairpin_design/operations/runtime/command-groups/pipeline\.yaml' "skill references pinned study pipeline"
require_pattern 'references/test-matrix\.md' "skill references test matrix for validation"
require_pattern 'composition_payload\.py' "skill routes composition payload source separately"
require_pattern 'output_guards\.py' "skill routes output guard source separately"
require_pattern 'materialized_outputs\.py' "skill routes artifact publication source separately"
require_pattern 'manifests\.py' "skill routes manifest writer source separately"
require_pattern 'developers\.openai\.com/api/docs/guides/prompt-engineering#coding' "skill records OpenAI Developers prompt-surface source"
require_pattern 'Start with input completeness, not study phase' "skill uses compiler-first routing"
require_pattern 'status command output' "skill keeps status as optional progress evidence"
require_pattern 'released-product Snapback' "skill preserves released-product boundary"
require_pattern 'scar-nick' "skill preserves scar-nick boundary language"
require_pattern 'S0=M' "skill preserves scar-compatible ligation invariant"
require_pattern 'generic linear ssDNA composition' "skill preserves composition route language"
require_pattern 'msd_design_reference_v1' "skill preserves MSD design-reference contract"
require_pattern 'msd_design_catalog_v1' "skill preserves MSD design-catalog contract"
require_pattern 'reference_index\.tsv' "skill preserves shallow reference index"
require_pattern 'flat .*references' "skill preserves flat reference bundle language"
require_pattern 'legacy.*assets' "skill preserves legacy layout refusal"
require_pattern 'stale.*references' "skill preserves stale reference refusal"
require_pattern 'one MSD unit|single-unit' "skill preserves single-unit materialization language"
require_pattern '5'\'' flank \+ left base' "skill names the MSD unit composition"
require_pattern 'Do not add `--repeat-count`|reject `--repeat-count`|does not expose `--repeat-count`' "skill records repeat-count footgun guard"
require_pattern 'sequence_manifest\.json' "skill preserves sequence manifest language"
require_pattern 'secondary_structure\.native\.png.*composition_overview\.svg|composition_overview\.svg.*secondary_structure\.native\.png' "skill preserves structure/review visual handoff"
require_pattern 'concrete .*sequence' "skill preserves missing-subcomponent fail-fast language"
require_pattern 'top-level `retron-msd`' "skill rejects top-level retron-msd leakage"
require_pattern 'Do not say "snapshot posture"' "skill rejects snapshot-posture output"
require_pattern 'four-base left/right basal spans' "skill preserves scar-nick projection boundary"
require_pattern 'YIU' "skill preserves YIU boundary language"
require_pattern 'harness-engineering' "skill routes harness work outward"
require_pattern 'code-change-discipline' "skill routes contract work outward"
require_pattern '\.agents/skills/retron-hairpin-study/SKILL\.md' "root AGENTS mention the skill" "$ROOT_AGENTS"
require_pattern '\.agents/skills/retron-hairpin-study/SKILL\.md' "cruncher AGENTS mentions the skill" "$CRUNCHER_AGENTS"
reject_pattern "$OLD_SKILL_NAME" "old skill route removed"
if grep -REq 'fallback' "$SKILL_FILE" "$REFERENCE_DIR" "$REPO_ROOT/docs/studies/retron_hairpin_design"; then
  fail "skill and Retron study docs avoid fallback language"
else
  pass "skill and Retron study docs avoid fallback language"
fi
reject_pattern 'whether the answer came from snapshot posture' "top-level skill no longer requires snapshot posture" "$SKILL_FILE"
reject_pattern 'current phase and next route' "top-level skill no longer reports phase by default" "$SKILL_FILE"
require_pattern 'Pair with `harness-engineering`' "skill explains harness pairing" "$SKILL_FILE"
require_pattern 'Pair with `code-change-discipline`' "skill explains code-change pairing" "$SKILL_FILE"
require_pattern 'Fresh/naive agent' "test matrix covers naive-agent discovery"
require_pattern 'Biopython reads the GenBank' "test matrix covers GenBank and reverse-complement handoff"

if [[ $failures -eq 0 ]]; then
  printf 'Audit finished with no failures.\n'
else
  printf 'Audit finished with %d failure(s).\n' "$failures"
  exit 1
fi
