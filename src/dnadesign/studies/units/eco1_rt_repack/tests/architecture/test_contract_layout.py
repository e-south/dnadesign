"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/architecture/test_contract_layout.py

Contract-layout and skill-frontmatter tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import yaml

from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root

_PACKAGE_ROOT = "src/dnadesign/studies/units/eco1_rt_repack"
_CONTRACT_ROOT_FILES = {
    "__init__.py",
    "artifact_chain.py",
    "common.py",
    "constants.py",
    "evidence_artifacts.py",
    "models.py",
    "profile.py",
    "suite.py",
}
_CONTRACT_SEMANTIC_PACKAGES = {"conservation", "contact_risk", "foldcheck", "masks", "sampling", "structure"}
_CONTRACT_CONSERVATION_FILES = {"__init__.py", "artifacts.py", "source_selection.py", "sources.py"}
_CONTRACT_CONTACT_RISK_FILES = {"__init__.py", "artifacts.py"}
_CONTRACT_FOLDCHECK_FILES = {"__init__.py", "request.py"}
_CONTRACT_MASK_FILES = {
    "__init__.py",
    "cases.py",
    "manual_artifacts.py",
    "rt_intervals.py",
    "set_artifacts.py",
    "source.py",
}
_CONTRACT_MASK_PACKAGES: set[str] = set()
_CONTRACT_SAMPLING_FILES = {
    "__init__.py",
    "artifacts.py",
    "candidate_table.py",
    "proteinmpnn_request.py",
    "sample_table.py",
}
_CONTRACT_SAMPLING_PACKAGES = {"thread_plan"}
_CONTRACT_THREAD_PLAN_FILES = {
    "__init__.py",
    "constants.py",
    "expected.py",
    "io.py",
    "report.py",
    "validation.py",
}
_CONTRACT_STRUCTURE_FILES = {
    "__init__.py",
    "artifacts.py",
    "authority.py",
    "contact_geometry.py",
    "preprocessing.py",
    "provenance.py",
}
_CONTRACT_TEST_ROOT_FILES = {
    "__init__.py",
    "test_phase_contracts.py",
    "test_source_contracts.py",
}
_CONTRACT_TEST_SEMANTIC_PACKAGES = {"conservation", "contact_risk", "foldcheck", "masks", "sampling", "structure"}
_CONTRACT_CONSERVATION_TEST_FILES = {"__init__.py", "test_sources.py"}
_CONTRACT_CONTACT_RISK_TEST_FILES = {"__init__.py", "test_artifacts.py"}
_CONTRACT_FOLDCHECK_TEST_FILES = {"__init__.py", "test_request.py"}
_CONTRACT_MASK_TEST_FILES = {
    "__init__.py",
    "test_cases.py",
    "test_rt_intervals.py",
    "test_source.py",
}
_CONTRACT_MASK_TEST_PACKAGES: set[str] = set()
_CONTRACT_SAMPLING_TEST_FILES = {
    "__init__.py",
    "test_candidate_table.py",
    "test_proteinmpnn_request.py",
    "test_sample_table.py",
}
_CONTRACT_SAMPLING_TEST_PACKAGES = {"thread_plan"}
_CONTRACT_THREAD_PLAN_TEST_FILES = {"__init__.py", "test_contract.py"}
_CONTRACT_STRUCTURE_TEST_FILES = {
    "__init__.py",
    "test_authority.py",
    "test_contact_geometry.py",
    "test_preprocessing.py",
}


def test_contract_package_routes_domain_contracts_through_semantic_subpackages() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/contracts"
    test_root = repo_root() / _PACKAGE_ROOT / "tests/contracts"

    assert sorted(path.name for path in source_root.glob("*.py")) == sorted(_CONTRACT_ROOT_FILES)
    assert sorted(path.name for path in source_root.iterdir() if path.is_dir() and path.name != "__pycache__") == (
        sorted(_CONTRACT_SEMANTIC_PACKAGES)
    )
    assert sorted(path.name for path in (source_root / "conservation").glob("*.py")) == sorted(
        _CONTRACT_CONSERVATION_FILES
    )
    assert sorted(path.name for path in (source_root / "contact_risk").glob("*.py")) == sorted(
        _CONTRACT_CONTACT_RISK_FILES
    )
    assert sorted(path.name for path in (source_root / "foldcheck").glob("*.py")) == sorted(_CONTRACT_FOLDCHECK_FILES)
    assert sorted(path.name for path in (source_root / "masks").glob("*.py")) == sorted(_CONTRACT_MASK_FILES)
    assert sorted(
        path.name for path in (source_root / "masks").iterdir() if path.is_dir() and path.name != "__pycache__"
    ) == sorted(_CONTRACT_MASK_PACKAGES)
    assert sorted(path.name for path in (source_root / "sampling").glob("*.py")) == sorted(_CONTRACT_SAMPLING_FILES)
    assert sorted(
        path.name for path in (source_root / "sampling").iterdir() if path.is_dir() and path.name != "__pycache__"
    ) == sorted(_CONTRACT_SAMPLING_PACKAGES)
    assert sorted(path.name for path in (source_root / "sampling/thread_plan").glob("*.py")) == sorted(
        _CONTRACT_THREAD_PLAN_FILES
    )
    assert sorted(path.name for path in (source_root / "structure").glob("*.py")) == sorted(_CONTRACT_STRUCTURE_FILES)
    assert sorted(path.name for path in test_root.glob("*.py")) == sorted(_CONTRACT_TEST_ROOT_FILES)
    assert sorted(path.name for path in test_root.iterdir() if path.is_dir() and path.name != "__pycache__") == sorted(
        _CONTRACT_TEST_SEMANTIC_PACKAGES
    )
    assert sorted(path.name for path in (test_root / "conservation").glob("*.py")) == sorted(
        _CONTRACT_CONSERVATION_TEST_FILES
    )
    assert sorted(path.name for path in (test_root / "contact_risk").glob("*.py")) == sorted(
        _CONTRACT_CONTACT_RISK_TEST_FILES
    )
    assert sorted(path.name for path in (test_root / "foldcheck").glob("*.py")) == sorted(
        _CONTRACT_FOLDCHECK_TEST_FILES
    )
    assert sorted(path.name for path in (test_root / "masks").glob("*.py")) == sorted(_CONTRACT_MASK_TEST_FILES)
    assert sorted(
        path.name for path in (test_root / "masks").iterdir() if path.is_dir() and path.name != "__pycache__"
    ) == sorted(_CONTRACT_MASK_TEST_PACKAGES)
    assert sorted(path.name for path in (test_root / "sampling").glob("*.py")) == sorted(_CONTRACT_SAMPLING_TEST_FILES)
    assert sorted(
        path.name for path in (test_root / "sampling").iterdir() if path.is_dir() and path.name != "__pycache__"
    ) == sorted(_CONTRACT_SAMPLING_TEST_PACKAGES)
    assert sorted(path.name for path in (test_root / "sampling/thread_plan").glob("*.py")) == sorted(
        _CONTRACT_THREAD_PLAN_TEST_FILES
    )
    assert sorted(path.name for path in (test_root / "structure").glob("*.py")) == sorted(
        _CONTRACT_STRUCTURE_TEST_FILES
    )


def test_eco1_source_and_test_modules_stay_within_line_budgets() -> None:
    package_root = repo_root() / _PACKAGE_ROOT

    oversized_source = [
        f"{path.relative_to(repo_root())}:{len(path.read_text(encoding='utf-8').splitlines())}"
        for path in package_root.rglob("*.py")
        if "tests" not in path.parts and len(path.read_text(encoding="utf-8").splitlines()) > 500
    ]
    oversized_tests = [
        f"{path.relative_to(repo_root())}:{len(path.read_text(encoding='utf-8').splitlines())}"
        for path in (package_root / "tests").rglob("*.py")
        if len(path.read_text(encoding="utf-8").splitlines()) > 200
    ]

    assert oversized_source == []
    assert oversized_tests == []


def test_contract_validator_cli_stays_thin() -> None:
    cli_path = repo_root() / _PACKAGE_ROOT / "operations/contract_validation.py"
    text = cli_path.read_text(encoding="utf-8")

    assert len(text.splitlines()) <= 80
    assert "validate_checked_in_contracts" in text
    assert "def validate_" not in text


def test_status_skill_uses_progressive_disclosure_frontmatter() -> None:
    skill_path = repo_root() / ".agents/skills/eco1-rt-repack-status/SKILL.md"
    text = skill_path.read_text(encoding="utf-8")
    frontmatter = yaml.safe_load(text.split("---", 2)[1])

    assert frontmatter["name"] == "eco1-rt-repack-status"
    assert len(frontmatter["description"]) <= 220
    assert "Do not use for another study or for family-level routing" in frontmatter["description"]
    assert "- [study-surfaces.md](references/study-surfaces.md)" in text
    assert "- [route-matrix.md](references/route-matrix.md)" in text
    assert "- [test-matrix.md](references/test-matrix.md)" in text
