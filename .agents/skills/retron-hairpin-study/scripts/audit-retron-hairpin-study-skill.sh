#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../../.." && pwd)"
SKILL_FILE="$ROOT_DIR/SKILL.md"
REFERENCE_DIR="$ROOT_DIR/references"
ROOT_AGENTS="$REPO_ROOT/AGENTS.md"
CRUNCHER_AGENTS="$REPO_ROOT/src/dnadesign/cruncher/AGENTS.md"
MSD_REGION_INGEST_CLI="$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/interfaces/cli/msd_region_ingest.py"
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
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/compiler/inputs/teto_retained_span_trim_tetr_pwm_elite_v1.spec.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/compiler/inputs/teto_retained_span_trim_ecoli_working_v1.spec.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/README.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/ontology/README.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/ontology/directions.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/ontology/payload_binding_sites.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/design_sets/README.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/design_sets/scar_nick_profile_panel_v1.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/design_sets/teto_retained_span_trim_tetr_pwm_elite_v1.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/design_sets/teto_retained_span_trim_ecoli_working_v1.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/deliverables/README.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/deliverables/teto_retained_span_trim_tetr_pwm_elite_v1.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/deliverables/teto_retained_span_trim_ecoli_working_v1.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/provenance/README.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/provenance/compiler_runs/README.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/provenance/compiler_runs/2026-05-18-msd-177-194.compile.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/provenance/compiler_runs/2026-06-20-teto-retained-span-trim-tetr-pwm-elite-v1.compile.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/provenance/materializations/README.md"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/provenance/materializations/2026-05-18-msd-177-194.single-unit.yaml"
require_file "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/provenance/materializations/2026-06-20-teto-retained-span-trim-tetr-pwm-elite-v1.single-unit.yaml"
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
require_file "$MSD_REGION_INGEST_CLI"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/interfaces/cli/review_outputs.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/catalog/compiler_spec.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/catalog/specs/__init__.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/catalog/specs/primitive_sources.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/catalog/specs/variant_metadata.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/catalog/msd_ids.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/catalog/registry.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/catalog/sequence_inputs.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/artifact_contracts/composition_payload.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/artifact_contracts/output_guards.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/artifact_contracts/materialized_outputs.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/artifact_contracts/manifests.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/artifact_contracts/layout.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/review_outputs/contracts/manifest.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/review_outputs/contracts/benchling_import.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/review_outputs/contracts/plan.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/review_outputs/contracts/review_variant_ids.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/review_outputs/handoff/benchling.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/review_outputs/handoff/contract.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/review_outputs/handoff/index.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/review_outputs/pwm/logo.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/review_outputs/pwm/sequence_rows.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/review_outputs/pwm/triptych.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/review_outputs/sequence/evidence.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/review_outputs/sequence/index.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/review_outputs/sequence/variant_identity.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/review_outputs/service.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/review_outputs/video/frame_naming.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/review_outputs/video/montage.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/review_outputs/video/stills.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/source_ingest/annotation_review.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/source_ingest/bundle_writer.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/source_ingest/compiler_spec_payload.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/source_ingest/comparison.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/source_ingest/feature_roles.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/source_ingest/genbank_bundle.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/source_ingest/genbank_utils.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/source_ingest/models.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/source_ingest/msd_region_genbank.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/source_ingest/pairing_segments.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/source_ingest/payload_binding.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/source_ingest/payload_binding_models.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/source_ingest/payload_binding_utils.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/source_ingest/payload_catalog.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/source_ingest/payload_motifs.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/source_ingest/payload_sites.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/source_ingest/record_normalization.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/source_ingest/variant_sources.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/compiler/test_cap_sources.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/compiler/test_msd_ids.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/compiler/test_cli_lint.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/compiler/test_msd_unit.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/compiler/test_cli_compile.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/compiler/test_materialization.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/compiler/test_boundaries.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/compiler/specs/test_teto_trim_metadata.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/cli/fixtures.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/cli/test_review_outputs.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/cli/test_review_outputs_text.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/package/test_generation.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/package/test_review_variant_ids.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/package/test_validation_failures.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/pwm/test_retention.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/handoff/test_benchling_import.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/video/test_montage.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/video/test_review_still_quality.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/support/cli.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/support/compiler_fixtures.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/support/pwm_fixtures.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/support/registry.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/support/review_ids.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/support/review_plans.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/support/review_outputs.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/support/viennarna.py"
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

if [[ -e "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests/compiler/test_msd_compiler.py" ]]; then
  fail "Retron compiler tests split into semantic lanes"
else
  pass "Retron compiler tests split into semantic lanes"
fi

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
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/design_sets/teto_retained_span_trim_tetr_pwm_elite_v1.yaml"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/deliverables/README.md"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/workbench/deliverables/teto_retained_span_trim_tetr_pwm_elite_v1.yaml"
  printf '\n'
  cat "$REPO_ROOT/docs/studies/retron_hairpin_design/compiler/inputs/teto_retained_span_trim_tetr_pwm_elite_v1.spec.yaml"
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
    "interfaces/cli/review_outputs.py": 120,
    "artifact_contracts/composition_payload.py": 450,
    "artifact_contracts/output_guards.py": 450,
    "artifact_contracts/materialized_outputs.py": 450,
    "artifact_contracts/manifests.py": 450,
    "review_outputs/contracts/manifest.py": 140,
    "review_outputs/contracts/benchling_import.py": 140,
    "review_outputs/contracts/plan.py": 180,
    "review_outputs/contracts/review_variant_ids.py": 130,
    "review_outputs/handoff/benchling.py": 190,
    "review_outputs/handoff/contract.py": 80,
    "review_outputs/handoff/index.py": 160,
    "review_outputs/pwm/logo.py": 200,
    "review_outputs/pwm/baserender_record.py": 170,
    "review_outputs/pwm/panel_labels.py": 50,
    "review_outputs/pwm/panel_metadata.py": 100,
    "review_outputs/pwm/retention.py": 240,
    "review_outputs/pwm/sequence_rows.py": 190,
    "review_outputs/pwm/trim_annotations.py": 60,
    "review_outputs/pwm/triptych.py": 140,
    "review_outputs/pwm/typography.py": 40,
    "review_outputs/pwm/visual_layers.py": 80,
    "review_outputs/sequence/evidence.py": 120,
    "review_outputs/sequence/index.py": 140,
    "review_outputs/sequence/variant_identity.py": 100,
    "review_outputs/service.py": 120,
    "review_outputs/video/frame_naming.py": 70,
    "review_outputs/video/montage.py": 170,
    "review_outputs/video/stills.py": 150,
    "catalog/cap_sources.py": 220,
    "catalog/compiler_spec.py": 450,
    "catalog/compiler_spec_io.py": 140,
    "catalog/specs/primitive_sources.py": 90,
    "catalog/specs/variant_metadata.py": 140,
    "catalog/sequence_inputs.py": 120,
    "source_ingest/annotation_review.py": 190,
    "source_ingest/bundle_writer.py": 130,
    "source_ingest/compiler_spec_payload.py": 100,
    "source_ingest/comparison.py": 230,
    "source_ingest/feature_roles.py": 110,
    "source_ingest/genbank_bundle.py": 190,
    "source_ingest/genbank_utils.py": 150,
    "source_ingest/models.py": 280,
    "source_ingest/msd_region_genbank.py": 50,
    "source_ingest/pairing_segments.py": 190,
    "source_ingest/payload_binding.py": 50,
    "source_ingest/payload_binding_models.py": 90,
    "source_ingest/payload_binding_utils.py": 110,
    "source_ingest/payload_catalog.py": 180,
    "source_ingest/payload_motifs.py": 150,
    "source_ingest/payload_sites.py": 220,
    "source_ingest/record_normalization.py": 270,
    "source_ingest/variant_sources.py": 180,
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
root_review_modules = sorted(path.name for path in (source_root / "review_outputs").glob("*.py"))
if root_review_modules != ["__init__.py", "service.py"]:
    violations.append(f"review_outputs root has flat modules: {root_review_modules}")
if list((source_root / "review_outputs").glob("pwm_*.py")):
    violations.append("review_outputs root contains flat pwm_*.py files")
if list((source_root / "review_outputs").glob("sequence_*.py")):
    violations.append("review_outputs root contains flat sequence_*.py files")
if (source_root / "review_outputs" / "clone_handoff_index.py").exists():
    violations.append("review_outputs/clone_handoff_index.py still exists")
if (source_root / "source_ingest" / "source_ingest.py").exists():
    violations.append("source_ingest/source_ingest.py should not collapse semantic modules")
if violations:
    raise SystemExit("; ".join(violations))
PY
then
  pass "Retron compiler source stays decomposed by responsibility"
else
  fail "Retron compiler source stays decomposed by responsibility"
fi

if uv run python - "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/tests" <<'PY'
from pathlib import Path
import sys

tests_root = Path(sys.argv[1])
budgets = {
    "compiler/test_cap_sources.py": 120,
    "compiler/test_msd_ids.py": 120,
    "compiler/test_cli_lint.py": 1000,
    "compiler/test_msd_unit.py": 120,
    "compiler/test_cli_compile.py": 280,
    "compiler/test_materialization.py": 900,
    "compiler/test_boundaries.py": 245,
    "compiler/specs/test_teto_trim_metadata.py": 140,
    "review_outputs/cli/fixtures.py": 70,
    "review_outputs/cli/test_review_outputs.py": 90,
    "review_outputs/cli/test_review_outputs_text.py": 70,
    "review_outputs/handoff/test_benchling_import.py": 110,
    "review_outputs/package/test_generation.py": 230,
    "review_outputs/package/test_review_variant_ids.py": 70,
    "review_outputs/package/test_validation_failures.py": 110,
    "review_outputs/pwm/test_retention.py": 100,
    "review_outputs/video/test_montage.py": 100,
    "review_outputs/video/test_review_still_quality.py": 110,
    "source_ingest/test_msd_region_genbank.py": 560,
    "support/cli.py": 40,
    "support/compiler_fixtures.py": 80,
    "support/pwm_fixtures.py": 70,
    "support/registry.py": 80,
    "support/review_ids.py": 40,
    "support/review_plans.py": 60,
    "support/review_outputs.py": 220,
    "support/viennarna.py": 100,
}

violations = []
for filename, budget in budgets.items():
    path = tests_root / filename
    line_count = len(path.read_text(encoding="utf-8").splitlines())
    if line_count > budget:
        violations.append(f"{filename} has {line_count} lines > {budget}")
if (tests_root / "compiler" / "test_msd_compiler.py").exists():
    violations.append("compiler/test_msd_compiler.py still exists")
root_review_tests = sorted(path.name for path in (tests_root / "review_outputs").glob("test_*.py"))
if root_review_tests:
    violations.append(f"review_outputs root has broad tests: {root_review_tests}")
if violations:
    raise SystemExit("; ".join(violations))
PY
then
  pass "Retron compiler tests stay decomposed by responsibility"
else
  fail "Retron compiler tests stay decomposed by responsibility"
fi

require_pattern 'docs/studies/README\.md' "skill routes through study records docs"
require_pattern 'docs/studies/retron_hairpin_design/record/status\.md' "skill references pinned study status"
require_pattern 'docs/studies/retron_hairpin_design/routes/README\.md' "skill references pinned study routes"
require_pattern 'references/msd-design-references\.md' "skill references MSD route detail"
require_pattern 'docs/studies/retron_hairpin_design/workbench/README\.md' "skill references workbench entrypoint"
require_pattern 'workbench/ontology' "skill references workbench ontology lane"
require_pattern 'workbench/deliverables' "skill references workbench deliverables lane"
require_pattern 'workbench/provenance' "skill references workbench provenance lane"
require_pattern 'scar_nick_profile_panel_v1\.yaml' "skill references workbench design set"
require_pattern 'docs/studies/retron_hairpin_design/contexts/cruncher/scar-nick-base-junction\.md' "skill references scar-nick base-junction context"
require_pattern 'docs/studies/retron_hairpin_design/contexts/composition/linear-ssdna-composition\.md' "skill references linear ssDNA composition handoff"
require_pattern 'docs/studies/retron_hairpin_design/compiler/catalog/msd_design_registry\.yaml' "skill references MSD design registry"
require_pattern 'docs/studies/retron_hairpin_design/compiler/catalog/msd_cap_sources\.yaml' "skill references cap source lookup"
require_pattern 'docs/studies/retron_hairpin_design/compiler/inputs/msd_design_hit_labels\.txt' "skill references MSD selected labels"
require_pattern 'docs/studies/retron_hairpin_design/compiler/inputs/msd_design_177_194_cap_sources_spec\.yaml' "skill references full cohort materialization spec"
require_pattern 'docs/studies/retron_hairpin_design/compiler/inputs/teto_retained_span_trim_tetr_pwm_elite_v1\.spec\.yaml' "skill references tetO trim compiler spec"
require_pattern 'teto_retained_span_trim_tetr_pwm_elite_v1\.yaml' "skill references tetO trim design set"
require_pattern 'teto_retained_span_trim_ecoli_working_v1' "skill references Eco1 tetO retained-span trim"
require_pattern 'payload_binding_sites\.yaml' "skill references payload binding ontology"
require_pattern 'pwm_trim_triptych' "skill references tetO PWM trim review panel"
require_pattern 'sequence_montage' "skill references tetO sequence review video"
require_pattern 'reviews/video/stills|semantic still' "skill references tetO semantic review stills"
require_pattern 'reverse-complement' "skill references review-output reverse-complement evidence"
require_pattern 'sequence_handoff' "skill references tetO GenBank handoff bundle"
require_pattern 'deliverable plan' "skill keeps review-output route plan-owned"
require_pattern 'payload-trim metadata|payload_trim_id' "skill preserves payload-trim metadata routing"
require_pattern 'WT Eco1' "skill preserves WT Eco1-only trim lane"
require_pattern 'docs/dev/plans/cross-tool/linear-ssdna-composition/2026-05-13-generic-linear-ssdna-composition.md' "skill references linear ssDNA dev spec"
require_pattern 'docs/exec-plans/completed/2026-05-13-generic-linear-ssdna-composition\.md' "skill references linear ssDNA implementation record"
require_pattern 'docs/exec-plans/active/2026-05-14-linear-ssdna-composition-hardening-followups\.md' "skill references linear ssDNA follow-up plan"
require_pattern 'docs/studies/retron_hairpin_design/operations/runtime/command-groups/pipeline\.yaml' "skill references pinned study pipeline"
require_pattern 'references/test-matrix\.md' "skill references test matrix for validation"
require_pattern 'composition_payload\.py' "skill routes composition payload source separately"
require_pattern 'output_guards\.py' "skill routes output guard source separately"
require_pattern 'materialized_outputs\.py' "skill routes artifact publication source separately"
require_pattern 'manifests\.py' "skill routes manifest writer source separately"
require_pattern 'review_outputs/service\.py' "skill routes review output service separately"
require_pattern 'source_ingest/msd_region_genbank\.py' "skill routes source-ingest public boundary separately"
require_pattern 'source_ingest/genbank_bundle\.py' "skill routes GenBank bundle parser separately"
require_pattern 'source_ingest/variant_sources\.py' "skill routes per-variant source manifest separately"
require_pattern 'source_ingest/annotation_review\.py' "skill routes annotation review notes separately"
require_pattern 'source_ingest/pairing_segments\.py' "skill routes pairing segment derivation separately"
require_pattern 'source_ingest/payload_catalog\.py' "skill routes payload catalog loading separately"
require_pattern 'source_ingest/payload_motifs\.py' "skill routes payload motif scoring separately"
require_pattern 'source_ingest/payload_sites\.py' "skill routes payload-site classification separately"
require_pattern 'source-dir' "MSD-region ingest CLI exposes per-variant source directory" "$MSD_REGION_INGEST_CLI"
reject_pattern 'source-genbank|replacement-genbank|write-variant-source-inputs' "MSD-region ingest CLI has no bulk migration options" "$MSD_REGION_INGEST_CLI"
require_pattern 'retired bulk source' "skill records retired bulk source as provenance only"
require_pattern 'review-outputs --deliverable-plan' "skill records explicit review-output command"
require_pattern 'test_cli_lint\.py' "skill records compiler lint test lane"
require_pattern 'test_materialization\.py' "skill records compiler materialization test lane"
require_pattern 'tests/review_outputs' "skill records review-output test lane"
require_pattern 'tests/support' "skill records shared compiler test fixtures"
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
if [[ -e "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/catalog/primitive_sources.py" ]]; then
  fail "primitive selector specs moved into semantic package"
else
  pass "primitive selector specs moved into semantic package"
fi
if [[ -e "$REPO_ROOT/src/dnadesign/studies/units/retron_hairpin_design/catalog/variant_metadata.py" ]]; then
  fail "variant metadata specs moved into semantic package"
else
  pass "variant metadata specs moved into semantic package"
fi
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
