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
visualization_root = repo_root / "src/dnadesign/aligner/msa/visualization"
visualization_tests_root = repo_root / "src/dnadesign/aligner/tests/msa/visualization"
msa_backend_root = repo_root / "src/dnadesign/aligner/msa/backends"
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

expected_materialization_primitives = {
    "atlas_semantic_profile",
    "biohub_esmc_sae_profile",
    "candidate_table",
    "structure",
    "structure_preprocessing",
    "contact",
    "contact_geometry",
    "contact_risk",
    "conservation",
    "conservation_alignments",
    "foldcheck_review",
    "foldcheck_report",
    "foldcheck_request",
    "manual_mask_authority",
    "mask_set",
    "proteinmpnn_request",
    "proteinmpnn_sample_ingest",
    "review_deliverables",
    "source_sequences",
    "thread_plan",
}
observed_materialization_primitives = {
    path.name for path in materialization_root.iterdir() if path.is_dir() and path.name != "__pycache__"
}
if observed_materialization_primitives != expected_materialization_primitives:
    problems.append(
        "materialization primitives must stay study-owned and semantic, observed "
        f"{sorted(observed_materialization_primitives)}"
    )

for primitive in sorted(expected_materialization_primitives):
    package = materialization_root / primitive
    if not package.is_dir():
        problems.append(f"missing materialization/{primitive}/ semantic package")
        continue
    for required_name in ("__init__.py", "__main__.py", "pipeline.py"):
        if not (package / required_name).is_file():
            problems.append(f"missing materialization/{primitive}/{required_name}")

contact_geometry_root = materialization_root / "contact_geometry"
expected_contact_geometry_files = {
    "__init__.py",
    "__main__.py",
    "cli.py",
    "constants.py",
    "models.py",
    "paths.py",
    "pipeline.py",
    "rows.py",
    "structure_io.py",
    "writer.py",
}
observed_contact_geometry_files = {path.name for path in contact_geometry_root.glob("*.py")}
if observed_contact_geometry_files != expected_contact_geometry_files:
    problems.append(
        "contact_geometry materialization must stay decomposed by constants, models, paths, rows, "
        f"structure_io, writer, and pipeline, observed {sorted(observed_contact_geometry_files)}"
    )
contact_geometry_pipeline_text = (contact_geometry_root / "pipeline.py").read_text(encoding="utf-8")
for forbidden_snippet in ("Bio.PDB", "pyarrow as pa", "np.stack", "MMCIFParser"):
    if forbidden_snippet in contact_geometry_pipeline_text:
        problems.append(f"contact_geometry pipeline reabsorbed implementation detail: {forbidden_snippet}")

proteinmpnn_request_root = materialization_root / "proteinmpnn_request"
expected_proteinmpnn_request_files = {
    "__init__.py",
    "__main__.py",
    "cli.py",
    "constants.py",
    "models.py",
    "pipeline.py",
}
observed_proteinmpnn_request_files = {path.name for path in proteinmpnn_request_root.glob("*.py")}
if observed_proteinmpnn_request_files != expected_proteinmpnn_request_files:
    problems.append(
        "proteinmpnn_request materialization must stay a thin Eco1 wrapper around constants, models, "
        f"and pipeline, observed {sorted(observed_proteinmpnn_request_files)}"
    )
proteinmpnn_pipeline_text = (proteinmpnn_request_root / "pipeline.py").read_text(encoding="utf-8")
for forbidden_snippet in ("pyarrow", "hashlib", "protein_mpnn_run.py", "parse_multiple_chains.py"):
    if forbidden_snippet in proteinmpnn_pipeline_text:
        problems.append(f"proteinmpnn_request pipeline reabsorbed implementation detail: {forbidden_snippet}")
if "dnadesign.thread.adapters.proteinmpnn" not in proteinmpnn_pipeline_text:
    problems.append("proteinmpnn_request pipeline must delegate generic request mechanics to dnadesign.thread")

thread_proteinmpnn_root = repo_root / "src/dnadesign/thread/adapters/proteinmpnn"
expected_thread_proteinmpnn_files = {
    "__init__.py",
    "execution.py",
    "execution_preflight.py",
    "hashing.py",
    "manifest.py",
    "models.py",
    "positions.py",
    "samples.py",
    "sidecars.py",
    "structure.py",
    "validation.py",
}
observed_thread_proteinmpnn_files = {path.name for path in thread_proteinmpnn_root.glob("*.py")}
if observed_thread_proteinmpnn_files != expected_thread_proteinmpnn_files:
    problems.append(
        "thread ProteinMPNN adapter must stay decomposed by generic request responsibility, observed "
        f"{sorted(observed_thread_proteinmpnn_files)}"
    )
for path in thread_proteinmpnn_root.glob("*.py"):
    text = path.read_text(encoding="utf-8").lower()
    if "eco1" in text or "ec86" in text or "mestre" in text or "wang" in text:
        problems.append(f"generic thread adapter leaked Eco1 study semantics: {path.relative_to(repo_root)}")

source_sequences_root = materialization_root / "source_sequences"
expected_source_sequence_files = {
    "__init__.py",
    "__main__.py",
    "cli.py",
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

for package_name in ("contracts", "provider_sources", "providers", "roster_cache", "sufficiency"):
    package = source_sequences_root / package_name
    if not package.is_dir():
        problems.append(f"missing source_sequences/{package_name}/ semantic package")
        continue
    if not (package / "__init__.py").is_file():
        problems.append(f"missing source_sequences/{package_name}/__init__.py")

for package_name in ("provider_sources", "roster_cache", "sufficiency"):
    package = source_sequences_root / package_name
    if package.is_dir() and not (package / "cli.py").is_file():
        problems.append(f"missing source_sequences/{package_name}/cli.py")

for pipeline_path in (
    source_sequences_root / "pipeline.py",
    source_sequences_root / "provider_sources" / "pipeline.py",
    source_sequences_root / "roster_cache" / "pipeline.py",
):
    if pipeline_path.exists() and "argparse" in pipeline_path.read_text(encoding="utf-8"):
        problems.append(f"pipeline owns CLI parsing instead of cli.py: {pipeline_path.relative_to(repo_root)}")

for dirname in ("contracts", "materialization"):
    if not (tests_root / dirname).is_dir():
        problems.append(f"missing tests/{dirname}/ mirrored package")

contracts_root = operations_root / "contracts"
expected_contract_root_files = {
    "__init__.py",
    "artifact_chain.py",
    "common.py",
    "constants.py",
    "evidence_artifacts.py",
    "models.py",
    "profile.py",
    "suite.py",
}
observed_contract_root_files = {path.name for path in contracts_root.glob("*.py")}
if observed_contract_root_files != expected_contract_root_files:
    problems.append(
        "operations/contracts root must stay shared-orchestration only, observed "
        f"{sorted(observed_contract_root_files)}"
    )

expected_contract_packages = {"conservation", "contact_risk", "foldcheck", "masks", "sampling", "structure"}
observed_contract_packages = {
    path.name for path in contracts_root.iterdir() if path.is_dir() and path.name != "__pycache__"
}
if observed_contract_packages != expected_contract_packages:
    problems.append(
        "operations/contracts domain validators must live in semantic packages, observed "
        f"{sorted(observed_contract_packages)}"
    )

expected_contract_package_files = {
    "conservation": {"__init__.py", "artifacts.py", "source_selection.py", "sources.py"},
    "contact_risk": {"__init__.py", "artifacts.py"},
    "foldcheck": {"__init__.py", "report.py", "request.py"},
    "masks": {
        "__init__.py",
        "cases.py",
        "manual_artifacts.py",
        "rt_intervals.py",
        "set_artifacts.py",
        "source.py",
    },
    "sampling": {"__init__.py", "artifacts.py", "candidate_table.py", "proteinmpnn_request.py", "sample_table.py"},
    "structure": {"__init__.py", "artifacts.py", "authority.py", "contact_geometry.py", "preprocessing.py", "provenance.py"},
}
for package_name, expected_files in sorted(expected_contract_package_files.items()):
    package = contracts_root / package_name
    if not package.is_dir():
        problems.append(f"missing operations/contracts/{package_name}/ semantic package")
        continue
    observed_files = {path.name for path in package.glob("*.py")}
    if observed_files != expected_files:
        problems.append(
            f"operations/contracts/{package_name} file set drifted, observed {sorted(observed_files)}"
        )

sampling_package_dirs = {
    path.name for path in (contracts_root / "sampling").iterdir() if path.is_dir() and path.name != "__pycache__"
}
if sampling_package_dirs != {"thread_plan"}:
    problems.append(
        "operations/contracts/sampling must keep thread-plan validation in a semantic package, observed "
        f"{sorted(sampling_package_dirs)}"
    )
expected_thread_plan_files = {"__init__.py", "constants.py", "expected.py", "io.py", "report.py", "validation.py"}
observed_thread_plan_files = {path.name for path in (contracts_root / "sampling/thread_plan").glob("*.py")}
if observed_thread_plan_files != expected_thread_plan_files:
    problems.append(
        "operations/contracts/sampling/thread_plan file set drifted, observed "
        f"{sorted(observed_thread_plan_files)}"
    )

flat_test_files = sorted(path.name for path in tests_root.glob("test_*.py"))
if flat_test_files:
    problems.append(f"flat study tests are not allowed at tests root: {flat_test_files}")

legacy_test = tests_root / "test_contract_validation.py"
if legacy_test.exists():
    problems.append(f"legacy flat test still exists: {legacy_test.relative_to(repo_root)}")

contract_test_root = tests_root / "contracts"
expected_contract_test_root_files = {"__init__.py", "test_phase_contracts.py", "test_source_contracts.py"}
observed_contract_test_root_files = {path.name for path in contract_test_root.glob("*.py")}
if observed_contract_test_root_files != expected_contract_test_root_files:
    problems.append(
        "tests/contracts root must stay shared-orchestration only, observed "
        f"{sorted(observed_contract_test_root_files)}"
    )

expected_contract_test_packages = {"conservation", "contact_risk", "foldcheck", "masks", "sampling", "structure"}
observed_contract_test_packages = {
    path.name for path in contract_test_root.iterdir() if path.is_dir() and path.name != "__pycache__"
}
if observed_contract_test_packages != expected_contract_test_packages:
    problems.append(
        "tests/contracts domain tests must mirror semantic contract packages, observed "
        f"{sorted(observed_contract_test_packages)}"
    )

expected_contract_test_package_files = {
    "conservation": {"__init__.py", "test_sources.py"},
    "contact_risk": {"__init__.py", "test_artifacts.py"},
    "foldcheck": {"__init__.py", "test_report.py", "test_request.py"},
    "masks": {
        "__init__.py",
        "test_cases.py",
        "test_rt_intervals.py",
        "test_source.py",
    },
    "sampling": {"__init__.py", "test_candidate_table.py", "test_proteinmpnn_request.py", "test_sample_table.py"},
    "structure": {"__init__.py", "test_authority.py", "test_contact_geometry.py", "test_preprocessing.py"},
}
for package_name, expected_files in sorted(expected_contract_test_package_files.items()):
    package = contract_test_root / package_name
    if not package.is_dir():
        problems.append(f"missing tests/contracts/{package_name}/ mirrored package")
        continue
    observed_files = {path.name for path in package.glob("*.py")}
    if observed_files != expected_files:
        problems.append(f"tests/contracts/{package_name} file set drifted, observed {sorted(observed_files)}")

sampling_test_package_dirs = {
    path.name for path in (contract_test_root / "sampling").iterdir() if path.is_dir() and path.name != "__pycache__"
}
if sampling_test_package_dirs != {"thread_plan"}:
    problems.append(
        "tests/contracts/sampling must mirror the thread-plan semantic package, observed "
        f"{sorted(sampling_test_package_dirs)}"
    )
expected_thread_plan_test_files = {"__init__.py", "test_contract.py"}
observed_thread_plan_test_files = {path.name for path in (contract_test_root / "sampling/thread_plan").glob("*.py")}
if observed_thread_plan_test_files != expected_thread_plan_test_files:
    problems.append(
        "tests/contracts/sampling/thread_plan file set drifted, observed "
        f"{sorted(observed_thread_plan_test_files)}"
    )

test_materialization_root = tests_root / "materialization"
flat_materialization_tests = sorted(
    path.name for path in test_materialization_root.glob("test_*.py")
)
if flat_materialization_tests:
    problems.append(f"flat materialization tests are not allowed: {flat_materialization_tests}")

observed_test_materialization_primitives = {
    path.name for path in test_materialization_root.iterdir() if path.is_dir() and path.name != "__pycache__"
}
if observed_test_materialization_primitives != expected_materialization_primitives:
    problems.append(
        "tests/materialization primitives must mirror study-owned materialization packages, observed "
        f"{sorted(observed_test_materialization_primitives)}"
    )

for primitive in sorted(expected_materialization_primitives):
    package = test_materialization_root / primitive
    if not package.is_dir():
        problems.append(f"missing tests/materialization/{primitive}/ mirrored package")
        continue
    if not (package / "test_materialization.py").is_file():
        problems.append(f"missing tests/materialization/{primitive}/test_materialization.py")

source_sequence_test_root = test_materialization_root / "source_sequences"
expected_source_sequence_test_files = {"__init__.py", "_fixtures.py", "_qc_fixtures.py", "test_materialization.py"}
observed_source_sequence_test_files = {path.name for path in source_sequence_test_root.glob("*.py")}
if observed_source_sequence_test_files != expected_source_sequence_test_files:
    problems.append(
        "tests/materialization/source_sequences root must stay package-level only, observed "
        f"{sorted(observed_source_sequence_test_files)}"
    )

expected_source_sequence_test_packages = {
    "contracts": "test_provider_accessions.py",
    "provider_sources": "test_materialization.py",
    "roster_cache": "test_materialization.py",
    "sufficiency": "test_sufficiency.py",
}
for package_name, required_test in expected_source_sequence_test_packages.items():
    package = source_sequence_test_root / package_name
    if not package.is_dir():
        problems.append(f"missing tests/materialization/source_sequences/{package_name}/ mirrored package")
        continue
    if not (package / "__init__.py").is_file():
        problems.append(f"missing tests/materialization/source_sequences/{package_name}/__init__.py")
    if not (package / required_test).is_file():
        problems.append(f"missing tests/materialization/source_sequences/{package_name}/{required_test}")

expected_visualization_root_files = {"__init__.py", "__main__.py", "cli.py"}
observed_visualization_root_files = {path.name for path in visualization_root.glob("*.py")}
if observed_visualization_root_files != expected_visualization_root_files:
    problems.append(
        "aligner.msa.visualization root must stay public-entrypoint only, observed "
        f"{sorted(observed_visualization_root_files)}"
    )

expected_visualization_packages = {"contracts", "materialization", "renderers"}
observed_visualization_packages = {
    path.name for path in visualization_root.iterdir() if path.is_dir() and path.name != "__pycache__"
}
if observed_visualization_packages != expected_visualization_packages:
    problems.append(
        "aligner.msa.visualization implementation must stay semantic, observed "
        f"{sorted(observed_visualization_packages)}"
    )

for package_name in sorted(expected_visualization_packages):
    package = visualization_root / package_name
    if not package.is_dir():
        problems.append(f"missing aligner/msa/visualization/{package_name}/ semantic package")
        continue
    if not (package / "__init__.py").is_file():
        problems.append(f"missing aligner/msa/visualization/{package_name}/__init__.py")

expected_visualization_test_files = {"__init__.py", "_fixtures.py", "test_exemplar_manifest.py", "test_materialization.py"}
observed_visualization_test_files = {path.name for path in visualization_tests_root.glob("*.py")}
if observed_visualization_test_files != expected_visualization_test_files:
    problems.append(
        "aligner MSA visualization tests must stay mirrored and bounded, observed "
        f"{sorted(observed_visualization_test_files)}"
    )

if not (msa_backend_root / "execution.py").is_file():
    problems.append("aligner.msa.backends.execution must own shared subprocess execution mechanics")

for backend_module_name in ("mafft.py", "clustalo.py"):
    backend_module = msa_backend_root / backend_module_name
    if not backend_module.is_file():
        problems.append(f"missing aligner/msa/backends/{backend_module_name}")
        continue
    backend_text = backend_module.read_text(encoding="utf-8")
    if "run_staged_backend_alignment" not in backend_text:
        problems.append(f"{backend_module_name} must call shared backend execution contract")
    for forbidden_snippet in ("write_bundle_manifest", "perf_counter", "uuid4", "hashlib"):
        if forbidden_snippet in backend_text:
            problems.append(
                f"{backend_module_name} reimplements backend execution detail: {forbidden_snippet}"
            )

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

oversized_visualization_source = [
    f"{path.relative_to(repo_root)}:{len(path.read_text(encoding='utf-8').splitlines())}"
    for path in visualization_root.rglob("*.py")
    if len(path.read_text(encoding="utf-8").splitlines()) > 320
]
if oversized_visualization_source:
    problems.append(f"visualization modules exceed 320-line IA budget: {oversized_visualization_source}")

oversized_visualization_tests = [
    f"{path.relative_to(repo_root)}:{len(path.read_text(encoding='utf-8').splitlines())}"
    for path in visualization_tests_root.rglob("*.py")
    if len(path.read_text(encoding="utf-8").splitlines()) > 320
]
if oversized_visualization_tests:
    problems.append(f"visualization tests exceed 320-line IA budget: {oversized_visualization_tests}")

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
require_file "$STUDY_ROOT/workbench/ontology/rt-annotation-tracks.yaml"
require_file "$STUDY_ROOT/workbench/ontology/manual-mask-authority.yaml"
require_file "$STUDY_ROOT/workbench/ontology/msa-exemplar-rows.yaml"
require_file "$STUDY_ROOT/workbench/ontology/msa-panel-spec.yaml"
require_file "$STUDY_ROOT/workbench/provenance/conservation-sources.yaml"
require_file "$REPO_ROOT/docs/dev/plans/cross-tool/thread/2026-06-19-eco1-rt-repack-thread.md"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/contracts"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/contact_risk"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/masks"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/masking"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/tests/contracts"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/tests/contracts/contact_risk"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/tests/contracts/masks"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/tests/materialization"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/structure"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact_risk"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation_alignments"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/manual_mask_authority"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/mask_set"
require_dir "$REPO_ROOT/src/dnadesign/aligner/msa/visualization"
require_dir "$REPO_ROOT/src/dnadesign/aligner/msa/visualization/contracts"
require_dir "$REPO_ROOT/src/dnadesign/aligner/msa/visualization/materialization"
require_dir "$REPO_ROOT/src/dnadesign/aligner/msa/visualization/renderers"
require_file "$REPO_ROOT/src/dnadesign/aligner/msa/backends/execution.py"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/contracts"
require_dir "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/sufficiency"
require_file "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/contracts/provider_accessions.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation/pipeline.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation_alignments/pipeline.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact_risk/cli.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact_risk/pipeline.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/manual_mask_authority/cli.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/manual_mask_authority/pipeline.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/mask_set/cli.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/mask_set/pipeline.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/masking/rows.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/conservation/artifacts.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/conservation/sources.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/contact_risk/artifacts.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/masks/source.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/masks/manual_artifacts.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/masks/rt_intervals.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/masks/set_artifacts.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/structure/artifacts.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/structure/authority.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/structure/preprocessing.py"
require_file "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/structure/provenance.py"
require_file "$REPO_ROOT/src/dnadesign/aligner/msa/visualization/materialization/pipeline.py"
require_file "$REPO_ROOT/src/dnadesign/aligner/msa/visualization/contracts/annotation_tracks.py"
require_file "$REPO_ROOT/src/dnadesign/aligner/msa/visualization/contracts/exemplar_rows.py"
require_file "$REPO_ROOT/src/dnadesign/aligner/msa/visualization/renderers/profile_qc.py"
require_file "$REPO_ROOT/src/dnadesign/aligner/msa/visualization/renderers/exemplar_windows.py"

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
require_pattern 'currently exposes generic ProteinMPNN' "skill keeps thread scope limited to implemented thread mechanics"
require_pattern 'rt_lnrna_sponging_construct_triage' "skill names downstream RT-lnRNA route boundary"
require_pattern 'No OPS provider is registered' "route matrix documents record-only status" "$REFERENCE_DIR/route-matrix.md"
require_pattern 'Tao et al\. 2026' "external source table records Tao method prior" "$REFERENCE_DIR/external-sources.md"
require_pattern 'Mestre et al\. 2020' "external source table records Mestre source ontology prior" "$REFERENCE_DIR/external-sources.md"
require_pattern 'Simon et al\. 2019' "external source table records Simon annotation prior" "$REFERENCE_DIR/external-sources.md"
require_pattern 'Wang et al\. 2022' "external source table records Wang Ec86 structure prior" "$REFERENCE_DIR/external-sources.md"
require_pattern '2026-06-23' "external source table records retrieval date" "$REFERENCE_DIR/external-sources.md"
require_pattern 'Must not be used as' "external source table records misuse boundaries" "$REFERENCE_DIR/external-sources.md"
require_pattern 'record-only' "status is explicitly record-only" "$STUDY_ROOT/record/status.md"
require_pattern 'planned' "artifact surface is planned" "$STUDY_ROOT/operations/contract/surfaces/artifacts.yaml"
require_pattern 'thread_artifact_chain_schema' "artifact surface includes artifact-chain schema" "$STUDY_ROOT/operations/contract/surfaces/artifacts.yaml"
require_pattern 'rt_lnrna_candidate_acceptance_schema' "artifact surface includes downstream acceptance schema" "$STUDY_ROOT/operations/contract/surfaces/artifacts.yaml"
require_pattern 'explicit_no_fallback' "profile fixture declares no-fallback backend policy" "$STUDY_ROOT/operations/contract/fixtures/thread/eco1_rt_v1.profile.yaml"
require_pattern 'ec86_clade9_conservation_v1' "conservation source contract declares Ec86 clade 9 profile" "$STUDY_ROOT/workbench/provenance/conservation-sources.yaml"
require_pattern 'mestre_s1_ec86_rt_clade9_after_qc' "conservation source contract requires Mestre clade 9 profile rows" "$STUDY_ROOT/workbench/provenance/conservation-sources.yaml"
require_pattern 'context_only_not_conservation_denominator' "conservation source contract keeps full Mestre roster out of scoring denominator" "$STUDY_ROOT/workbench/provenance/conservation-sources.yaml"
require_pattern 'no_silent_backend_fallback' "conservation source contract rejects silent alignment backend fallback" "$STUDY_ROOT/workbench/provenance/conservation-sources.yaml"
require_pattern 'ec86_iia3_cluster42_1_conservation_v1' "conservation source contract declares Eco1-like profile" "$STUDY_ROOT/workbench/provenance/conservation-sources.yaml"
require_pattern 'accession_patterns' "conservation source contract declares provider accession patterns" "$STUDY_ROOT/workbench/provenance/conservation-sources.yaml"
require_pattern 'materialization/conservation/' "study surfaces route conservation materializer package" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'materialization/conservation_alignments/' "study surfaces route conservation alignment package" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'manual-mask-authority.yaml' "study surfaces route manual mask-authority ontology" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'materialization/manual_mask_authority/' "study surfaces route manual mask-authority materializer package" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'wang_et_al_2022_ec86_cryoem_structure_priors' "manual mask authority source records Wang/Ec86 structural priors" "$STUDY_ROOT/workbench/ontology/manual-mask-authority.yaml"
require_pattern 'candidate_prior_not_mask_authoritative' "manual mask authority keeps Wang interface residues as candidate priors" "$STUDY_ROOT/workbench/ontology/manual-mask-authority.yaml"
require_pattern 'rt7_interval' "manual mask authority source materializes RT1-RT7 interval spans" "$STUDY_ROOT/workbench/ontology/manual-mask-authority.yaml"
require_pattern 'materialization/mask_set/' "study surfaces route mask-set materializer package" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'materialization/contact_risk/' "study surfaces route contact-risk materializer package" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'operations/contracts/contact_risk/' "study surfaces route contact-risk contract package" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'eco1_rt_clade9_plurality25_direct_contact5a_v1' "study surfaces describe current simple mask rule" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'shared structure-provenance hash' "study surfaces route structure hash-closure contract" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'operations/masking/' "study surfaces route shared mask-row algebra package" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'aligner/msa/visualization' "study surfaces route generic MSA visualization API" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'rt-annotation-tracks.yaml' "study surfaces route Eco1 annotation tracks" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'msa-exemplar-rows.yaml' "study surfaces route Eco1 exemplar rows" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'msa-panel-spec.yaml' "study surfaces route Eco1 panel spec" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'source_sequences/contracts/' "study surfaces route source-sequence contract package" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'source_sequences/sufficiency/' "study surfaces route source-sequence sufficiency package" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'provider accession policy' "study surfaces route provider accession contract policy" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'operations/contracts/conservation/' "study surfaces route conservation contract package" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'operations/contracts/structure/' "study surfaces route structure contract package" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'operations/contracts/sampling/' "study surfaces route sampling contract package" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'operations/materialization/thread_plan/' "study surfaces route thread-plan materializer package" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'operations/materialization/proteinmpnn_request/' "study surfaces route Eco1 ProteinMPNN request wrapper" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'operations/materialization/proteinmpnn_sample_ingest/' "study surfaces route Eco1 ProteinMPNN sample-ingest wrapper" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'operations/materialization/candidate_table/' "study surfaces route Eco1 candidate-table wrapper" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'operations/materialization/foldcheck_report/' "study surfaces route Eco1 fold-check report wrapper" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'operations/materialization/foldcheck_review/' "study surfaces route Eco1 fold-check review wrapper" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'operations/materialization/review_deliverables/' "study surfaces route Eco1 review deliverables wrapper" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'operations/materialization/biohub_esmc_sae_profile/' "study surfaces route Eco1 Biohub ESMC wrapper" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'src/dnadesign/thread/adapters/proteinmpnn/' "study surfaces route generic ProteinMPNN request adapter" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'src/dnadesign/thread/adapters/colabfold/' "study surfaces route generic ColabFold output normalizer" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'src/dnadesign/thread/adapters/biohub_esmc/' "study surfaces route generic Biohub ESMC adapter" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'src/dnadesign/thread/candidates/' "study surfaces route generic candidate-table builder" "$REFERENCE_DIR/study-surfaces.md"
require_pattern 'not a hidden run-all pipeline' "command-group README rejects hidden run-all execution" "$STUDY_ROOT/operations/runtime/command-groups/README.md"
require_pattern 'conservation_provider_sources' "pipeline records provider-source command lane" "$STUDY_ROOT/operations/runtime/command-groups/pipeline.yaml"
require_pattern 'conservation_roster_cache' "pipeline records roster-cache command lane" "$STUDY_ROOT/operations/runtime/command-groups/pipeline.yaml"
require_pattern 'conservation_source_sufficiency' "pipeline records source-sufficiency command lane" "$STUDY_ROOT/operations/runtime/command-groups/pipeline.yaml"
require_pattern 'conservation_alignments' "pipeline records alignment command lane" "$STUDY_ROOT/operations/runtime/command-groups/pipeline.yaml"
require_pattern 'contact_risk_profile' "pipeline records contact-risk command lane" "$STUDY_ROOT/operations/runtime/command-groups/pipeline.yaml"
require_pattern 'materialized_simple_mask_set' "sampling readiness gates thread_plan behind validated simple mask" "$STUDY_ROOT/operations/contract/readiness/checks/sampling_plan.yaml"
require_pattern 'required_policy_id: eco1_rt_clade9_plurality25_direct_contact5a_v1' "sampling readiness records simple mask policy id" "$STUDY_ROOT/operations/contract/readiness/checks/sampling_plan.yaml"
require_pattern 'non_fixed_missing_backbone' "sampling readiness records terminal missing-backbone class" "$STUDY_ROOT/operations/contract/readiness/checks/sampling_plan.yaml"
require_pattern 'dnadesign.studies.units.eco1_rt_repack.operations.materialization.thread_plan' "pipeline records thread-plan materializer command" "$STUDY_ROOT/operations/runtime/command-groups/pipeline.yaml"
require_pattern 'dnadesign.studies.units.eco1_rt_repack.operations.materialization.proteinmpnn_request' "pipeline records ProteinMPNN request materializer command" "$STUDY_ROOT/operations/runtime/command-groups/pipeline.yaml"
require_pattern 'dnadesign.studies.units.eco1_rt_repack.operations.materialization.proteinmpnn_sample_ingest' "pipeline records ProteinMPNN sample-ingest materializer command" "$STUDY_ROOT/operations/runtime/command-groups/pipeline.yaml"
require_pattern 'dnadesign.studies.units.eco1_rt_repack.operations.materialization.candidate_table' "pipeline records candidate-table materializer command" "$STUDY_ROOT/operations/runtime/command-groups/pipeline.yaml"
require_pattern 'dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_report' "pipeline records fold-check report materializer command" "$STUDY_ROOT/operations/runtime/command-groups/pipeline.yaml"
require_pattern 'dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review' "pipeline records fold-check review materializer command" "$STUDY_ROOT/operations/runtime/command-groups/pipeline.yaml"
require_pattern 'dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables' "pipeline records review deliverables materializer command" "$STUDY_ROOT/operations/runtime/command-groups/pipeline.yaml"
require_pattern 'dnadesign.studies.units.eco1_rt_repack.operations.materialization.biohub_esmc_sae_profile' "pipeline records Biohub ESMC materializer command" "$STUDY_ROOT/operations/runtime/command-groups/pipeline.yaml"
require_pattern 'phase1_thread_contract' "pipeline records Phase 1 validation command" "$STUDY_ROOT/operations/runtime/command-groups/pipeline.yaml"
require_pattern 'phase2_real_backend_ingest' "pipeline records Phase 2 validation command" "$STUDY_ROOT/operations/runtime/command-groups/pipeline.yaml"
require_pattern 'presence-only check' "command-group README documents Phase 1 hash closure" "$STUDY_ROOT/operations/runtime/command-groups/README.md"
require_absent 'argparse' "contact-risk pipeline keeps CLI parsing out of pipeline" "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact_risk/pipeline.py"
require_absent 'argparse' "manual mask pipeline keeps CLI parsing out of pipeline" "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/manual_mask_authority/pipeline.py"
require_absent 'argparse' "mask-set pipeline keeps CLI parsing out of pipeline" "$REPO_ROOT/src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/mask_set/pipeline.py"
require_pattern 'reject_as_target_without_declared_substitution' "conservation source contract rejects target drift" "$STUDY_ROOT/workbench/provenance/conservation-sources.yaml"
placeholder_pattern='TO''DO|\[''TO''DO'
require_absent "$placeholder_pattern" "skill contains no initializer placeholders"

if [[ "$failures" -ne 0 ]]; then
  printf 'Skill audit failed with %s failure(s).\n' "$failures" >&2
  exit 1
fi

printf 'Eco1 RT repack status skill audit passed.\n'
