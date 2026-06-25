"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/architecture/test_source_sequence_layout.py

Source-sequence package-layout regression tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root

_PACKAGE_ROOT = "src/dnadesign/studies/units/eco1_rt_repack"
_SOURCE_SEQUENCE_ROOT_FILES = {
    "__init__.py",
    "__main__.py",
    "cli.py",
    "io.py",
    "issues.py",
    "manifest.py",
    "paths.py",
    "pipeline.py",
}
_SOURCE_SEQUENCE_PACKAGES = {"contracts", "provider_sources", "providers", "roster_cache", "sufficiency"}
_SOURCE_SEQUENCE_TEST_ROOT_FILES = {"__init__.py", "_fixtures.py", "_qc_fixtures.py", "test_materialization.py"}
_SOURCE_SEQUENCE_TEST_PACKAGES = {
    "contracts": "test_provider_accessions.py",
    "provider_sources": "test_materialization.py",
    "roster_cache": "test_materialization.py",
    "sufficiency": "test_sufficiency.py",
}


def test_source_sequence_stack_uses_semantic_subpackages() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/materialization/source_sequences"

    assert sorted(path.name for path in source_root.glob("*.py")) == sorted(_SOURCE_SEQUENCE_ROOT_FILES)
    for package in sorted(_SOURCE_SEQUENCE_PACKAGES):
        assert (source_root / package).is_dir()
        assert (source_root / package / "__init__.py").is_file()
    for command_package in ("provider_sources", "roster_cache", "sufficiency"):
        assert (source_root / command_package / "cli.py").is_file()
        assert "argparse" not in (source_root / command_package / "pipeline.py").read_text(encoding="utf-8")
    assert "argparse" not in (source_root / "pipeline.py").read_text(encoding="utf-8")


def test_source_sequence_tests_mirror_semantic_subpackages() -> None:
    test_root = repo_root() / _PACKAGE_ROOT / "tests/materialization/source_sequences"

    assert sorted(path.name for path in test_root.glob("*.py")) == sorted(_SOURCE_SEQUENCE_TEST_ROOT_FILES)
    for package, test_file in sorted(_SOURCE_SEQUENCE_TEST_PACKAGES.items()):
        assert (test_root / package).is_dir()
        assert (test_root / package / "__init__.py").is_file()
        assert (test_root / package / test_file).is_file()
