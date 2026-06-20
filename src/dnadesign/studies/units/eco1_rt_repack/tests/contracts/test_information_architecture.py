"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/contracts/test_information_architecture.py

Information-architecture regression tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import yaml

from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root

_PACKAGE_ROOT = "src/dnadesign/studies/units/eco1_rt_repack"
_ALLOWED_OPERATION_ROOT_FILES = {"__init__.py", "contract_validation.py"}
_ALLOWED_TEST_ROOT_FILES = {"__init__.py", "_helpers.py"}
_MATERIALIZATION_PRIMITIVES = {"structure", "contact", "conservation", "source_sequences"}


def test_operations_root_stays_entrypoint_only() -> None:
    operations_root = repo_root() / _PACKAGE_ROOT / "operations"

    assert (operations_root / "contracts").is_dir()
    assert (operations_root / "materialization").is_dir()
    assert sorted(path.name for path in operations_root.glob("*.py")) == sorted(_ALLOWED_OPERATION_ROOT_FILES)


def test_tests_mirror_semantic_source_packages() -> None:
    tests_root = repo_root() / _PACKAGE_ROOT / "tests"

    assert (tests_root / "contracts").is_dir()
    assert (tests_root / "materialization").is_dir()
    assert sorted(path.name for path in tests_root.glob("*.py")) == sorted(_ALLOWED_TEST_ROOT_FILES)
    assert not list(tests_root.glob("test_*.py"))


def test_materialization_primitives_are_semantic_packages() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/materialization"
    test_root = repo_root() / _PACKAGE_ROOT / "tests/materialization"

    assert sorted(path.name for path in source_root.glob("*.py")) == ["__init__.py"]
    assert sorted(path.name for path in test_root.glob("*.py")) == ["__init__.py"]
    for primitive in sorted(_MATERIALIZATION_PRIMITIVES):
        source_package = source_root / primitive
        test_package = test_root / primitive
        assert source_package.is_dir()
        assert test_package.is_dir()
        assert (source_package / "__init__.py").is_file()
        assert (source_package / "__main__.py").is_file()
        assert (source_package / "pipeline.py").is_file()
        assert (test_package / "__init__.py").is_file()
        assert (test_package / "test_materialization.py").is_file()


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
