#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../../.." && pwd)"
SKILL_FILE="$ROOT_DIR/SKILL.md"
REFERENCE_DIR="$ROOT_DIR/references"
STUDY_ROOT="$REPO_ROOT/docs/studies/eco1_rt_repack"
failures=0

pass() { printf 'PASS: %s\n' "$1"; }
fail() { printf 'FAIL: %s\n' "$1"; failures=$((failures + 1)); }

require_file() {
  local path="$1"
  [[ -f "$path" ]] && pass "found ${path#$REPO_ROOT/}" || fail "missing file $path"
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] && pass "found directory ${path#$REPO_ROOT/}" || fail "missing directory $path"
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
if len(description) > 220:
    raise SystemExit(f"frontmatter description exceeds progressive-disclosure target: {len(description)} > 220")
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

require_source_information_architecture() {
  if uv run python - "$REPO_ROOT" <<'PY'
from pathlib import Path
import sys

repo_root = Path(sys.argv[1])
package_root = repo_root / "src/dnadesign/studies/units/eco1_rt_repack"
operations_root = package_root / "operations"
tests_root = package_root / "tests"
problems: list[str] = []

expected_operation_dirs = {"contracts", "materialization"}
for dirname in sorted(expected_operation_dirs):
    if not (operations_root / dirname).is_dir():
        problems.append(f"missing operations/{dirname}/ semantic package")

flat_operation_files = sorted(
    path.name
    for path in operations_root.glob("*.py")
    if path.name not in {"__init__.py", "contract_validation.py"}
)
if flat_operation_files:
    problems.append(f"flat operation modules are not allowed: {flat_operation_files}")

for stale_path in (
    operations_root / "structure_materialization.py",
    operations_root / "contact_materialization.py",
    operations_root / "conservation_materialization.py",
    operations_root / "runtime_artifacts.py",
):
    if stale_path.exists():
        problems.append(f"stale flat module still exists: {stale_path.relative_to(repo_root)}")

materialization_root = operations_root / "materialization"
flat_materializers = sorted(path.name for path in materialization_root.glob("*.py") if path.name != "__init__.py")
if flat_materializers:
    problems.append(f"flat materialization primitive modules are not allowed: {flat_materializers}")

for primitive in ("structure", "contact", "conservation", "source_sequences"):
    package = materialization_root / primitive
    if not package.is_dir():
        problems.append(f"missing materialization/{primitive}/ semantic package")
        continue
    for required_name in ("__init__.py", "__main__.py", "pipeline.py"):
        if not (package / required_name).is_file():
            problems.append(f"missing materialization/{primitive}/{required_name}")

source_sequences_root = materialization_root / "source_sequences"
expected_source_sequence_files = {
    "__init__.py",
    "__main__.py",
    "io.py",
    "issues.py",
    "manifest.py",
    "paths.py",
    "pipeline.py",
}
observed_source_sequence_files = {path.name for path in source_sequences_root.glob("*.py")}
if observed_source_sequence_files != expected_source_sequence_files:
    problems.append(
        "source_sequences root must stay entrypoint/shared-utils only, observed "
        f"{sorted(observed_source_sequence_files)}"
    )

for package_name in ("contracts", "providers", "roster_cache", "sufficiency"):
    package = source_sequences_root / package_name
    if not package.is_dir():
        problems.append(f"missing source_sequences/{package_name}/ semantic package")
        continue
    if not (package / "__init__.py").is_file():
        problems.append(f"missing source_sequences/{package_name}/__init__.py")

for dirname in ("contracts", "materialization"):
    if not (tests_root / dirname).is_dir():
        problems.append(f"missing tests/{dirname}/ mirrored package")

flat_test_files = sorted(path.name for path in tests_root.glob("test_*.py"))
if flat_test_files:
    problems.append(f"flat study tests are not allowed at tests root: {flat_test_files}")

legacy_test = tests_root / "test_contract_validation.py"
if legacy_test.exists():
    problems.append(f"legacy flat test still exists: {legacy_test.relative_to(repo_root)}")

test_materialization_root = tests_root / "materialization"
flat_materialization_tests = sorted(
    path.name for path in test_materialization_root.glob("test_*.py")
)
if flat_materialization_tests:
    problems.append(f"flat materialization tests are not allowed: {flat_materialization_tests}")

for primitive in ("structure", "contact", "conservation", "source_sequences"):
    package = test_materialization_root / primitive
    if not package.is_dir():
        problems.append(f"missing tests/materialization/{primitive}/ mirrored package")
        continue
    if not (package / "test_materialization.py").is_file():
        problems.append(f"missing tests/materialization/{primitive}/test_materialization.py")

cli_lines = len((operations_root / "contract_validation.py").read_text(encoding="utf-8").splitlines())
if cli_lines > 80:
    problems.append(f"contract_validation.py must stay CLI-thin, observed {cli_lines} lines")

oversized_source = [
    f"{path.relative_to(repo_root)}:{len(path.read_text(encoding='utf-8').splitlines())}"
    for path in package_root.rglob("*.py")
    if "tests" not in path.parts and len(path.read_text(encoding="utf-8").splitlines()) > 500
]
if oversized_source:
    problems.append(f"source modules exceed 500-line IA budget: {oversized_source}")

oversized_tests = [
    f"{path.relative_to(repo_root)}:{len(path.read_text(encoding='utf-8').splitlines())}"
    for path in tests_root.rglob("*.py")
    if len(path.read_text(encoding="utf-8").splitlines()) > 200
]
if oversized_tests:
    problems.append(f"test modules exceed 200-line IA budget: {oversized_tests}")

if problems:
    raise SystemExit("\n".join(problems))
PY
  then
    pass "source and test information architecture stays semantic and bounded"
  else
    fail "source and test information architecture stays semantic and bounded"
  fi
}

require_file "$SKILL_FILE"
require_file "$REFERENCE_DIR/study-surfaces.md"
require_file "$REFERENCE_DIR/route-matrix.md"
require_file "$REFERENCE_DIR/refresh-loop.md"
require_file "$REFERENCE_DIR/external-sources.md"
require_file "$REFERENCE_DIR/test-matrix.md"
require_file "$STUDY_ROOT/README.md"
require_file "$STUDY_ROOT/routes/README.md"
require_file "$STUDY_ROOT/record/status.md"
require_file "$STUDY_ROOT/record/datasets.yaml"
require_file "$STUDY_ROOT/record/campaign.yaml"
require_file "$STUDY_ROOT/operations/ops.study.yaml"
require_file "$STUDY_ROOT/operations/contract/readiness/checks/thread_profile.yaml"
require_file "$STUDY_ROOT/operations/contract/readiness/checks/structure_authority.yaml"
require_file "$STUDY_ROOT/operations/contract/readiness/checks/mask_contract.yaml"
require_file "$STUDY_ROOT/operations/contract/readiness/checks/sampling_plan.yaml"
require_file "$STUDY_ROOT/operations/contract/readiness/checks/candidate_handoff.yaml"
require_file "$STUDY_ROOT/operations/contract/readiness/checks/foldcheck_runtime.yaml"
require_file "$STUDY_ROOT/operations/contract/readiness/checks/assembly_feasibility.yaml"
require_file "$STUDY_ROOT/operations/contract/readiness/checks/downstream_rt_lnrna_handoff.yaml"
require_file "$STUDY_ROOT/operations/contract/schemas/eco1-rt-profile.schema.yaml"
require_file "$STUDY_ROOT/operations/contract/schemas/thread-artifact-chain.schema.yaml"
require_file "$STUDY_ROOT/operations/contract/schemas/thread-candidate-handoff.schema.yaml"
require_file "$STUDY_ROOT/operations/contract/schemas/rt-lnrna-candidate-acceptance.schema.yaml"
require_file "$STUDY_ROOT/contexts/implementation-roadmap.md"
require_file "$STUDY_ROOT/contexts/msa-method.md"
require_file "$STUDY_ROOT/contexts/residue-mask-policy.md"
require_file "$STUDY_ROOT/contexts/fold-validation-policy.md"
require_file "$STUDY_ROOT/workbench/provenance/conservation-sources.yaml"
require_file "$REPO_ROOT/docs/dev/plans/cross-tool/thread/2026-06-19-eco1-rt-repack-thread.md"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/contracts"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/tests/contracts"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/tests/materialization"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/structure"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/contracts"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/sufficiency"
require_file "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation/pipeline.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/conservation_artifacts.py"

require_frontmatter_yaml
require_source_information_architecture
require_section "## Required Deliverables"
require_section "## Trigger Tests"
require_pattern '^name: eco1-rt-repack-status$' "skill name is study-specific"
require_pattern 'eco1_rt_repack' "skill anchors the concrete study id"
require_pattern 'thread' "skill routes planned thread surfaces"
require_pattern 'structure_authority' "skill names structure authority readiness gate"
require_pattern 'mask_contract' "skill names mask contract readiness gate"
require_pattern 'sampling_plan' "skill names sampling plan readiness gate"
require_pattern 'foldcheck_runtime' "skill names fold-check readiness gate"
require_pattern 'assembly_feasibility' "skill names assembly feasibility readiness gate"
require_pattern 'implementation-roadmap' "skill names implementation roadmap surface"
require_pattern 'msa-method' "skill names MSA method surface"
require_pattern 'Do not use for another study or for family-level routing' "skill rejects family-level routing"
require_pattern 'Do not generalize it to another study' "skill guardrail rejects cross-study reuse"
require_pattern 'ec86kit target sequence hash' "skill guardrail rejects silent MSA target drift"
require_pattern 'Do not imply that `src/dnadesign/thread/` exists' "skill keeps planned thread separate from implementation"
require_pattern 'rt_lnrna_sponging_construct_triage' "skill names downstream RT-lnRNA route boundary"
require_pattern 'No OPS provider is registered' "route matrix documents record-only status" "$REFERENCE_DIR/route-matrix.md"
require_pattern 'record-only' "status is explicitly record-only" "$STUDY_ROOT/record/status.md"
require_pattern 'planned' "artifact surface is planned" "$STUDY_ROOT/operations/contract/surfaces/artifacts.yaml"
require_pattern 'thread_artifact_chain_schema' "artifact surface includes artifact-chain schema" "$STUDY_ROOT/operations/contract/surfaces/artifacts.yaml"
require_pattern 'rt_lnrna_candidate_acceptance_schema' "artifact surface includes downstream acceptance schema" "$STUDY_ROOT/operations/contract/surfaces/artifacts.yaml"
require_pattern 'explicit_no_fallback' "profile fixture declares no-fallback backend policy" "$STUDY_ROOT/operations/contract/fixtures/thread/eco1_rt_v1.profile.yaml"
require_pattern 'broad_retron_rt' "conservation source contract declares broad profile" "$STUDY_ROOT/workbench/provenance/conservation-sources.yaml"
require_pattern 'eco1_like_retron_rt' "conservation source contract declares Eco1-like profile" "$STUDY_ROOT/workbench/provenance/conservation-sources.yaml"
require_pattern 'materialization/conservation/' "study surfaces route conservation materializer package" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'source_sequences/contracts/' "study surfaces route source-sequence contract package" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'source_sequences/sufficiency/' "study surfaces route source-sequence sufficiency package" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'conservation_artifacts.py' "study surfaces route conservation artifact validator" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'reject_as_target_without_declared_substitution' "conservation source contract rejects target drift" "$STUDY_ROOT/workbench/provenance/conservation-sources.yaml"
placeholder_pattern='TO''DO|\[''TO''DO'
require_absent "$placeholder_pattern" "skill contains no initializer placeholders"

if [[ "$failures" -ne 0 ]]; then
  printf 'Skill audit failed with %s failure(s).\n' "$failures" >&2
  exit 1
fi

printf 'Eco1 RT repack status skill audit passed.\n'
