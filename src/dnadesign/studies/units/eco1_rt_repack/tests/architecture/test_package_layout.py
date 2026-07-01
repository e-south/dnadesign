"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/architecture/test_package_layout.py

Package-layout regression tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root

_PACKAGE_ROOT = "src/dnadesign/studies/units/eco1_rt_repack"
_ALLOWED_OPERATION_ROOT_FILES = {"__init__.py", "contract_validation.py"}
_ALLOWED_TEST_ROOT_FILES = {"__init__.py", "_helpers.py"}
_OPERATION_SEMANTIC_PACKAGES = {"contracts", "masking", "materialization"}
_TEST_SEMANTIC_PACKAGES = {"architecture", "contracts", "masking", "materialization"}
_MASKING_ROOT_FILES = {"__init__.py", "rows.py"}
_MASKING_TEST_ROOT_FILES = {"__init__.py", "test_rows.py"}
_MATERIALIZATION_PRIMITIVES = {
    "atlas_semantic_profile",
    "biohub_esmc_sae_profile",
    "biohub_esmc_wt_mutation_scoring",
    "candidate_table",
    "contact",
    "contact_geometry",
    "contact_risk",
    "conservation",
    "conservation_alignments",
    "design_classes",
    "foldcheck_review",
    "foldcheck_report",
    "foldcheck_request",
    "manual_mask_authority",
    "mask_set",
    "proteinmpnn_request",
    "proteinmpnn_sample_ingest",
    "review_deliverables",
    "source_sequences",
    "structure",
    "structure_preprocessing",
    "thread_plan",
}


def test_operations_root_stays_entrypoint_only() -> None:
    operations_root = repo_root() / _PACKAGE_ROOT / "operations"

    assert (operations_root / "contracts").is_dir()
    assert (operations_root / "masking").is_dir()
    assert (operations_root / "materialization").is_dir()
    assert sorted(path.name for path in operations_root.glob("*.py")) == sorted(_ALLOWED_OPERATION_ROOT_FILES)
    assert sorted(path.name for path in operations_root.iterdir() if path.is_dir() and path.name != "__pycache__") == (
        sorted(_OPERATION_SEMANTIC_PACKAGES)
    )


def test_tests_mirror_semantic_source_packages() -> None:
    tests_root = repo_root() / _PACKAGE_ROOT / "tests"

    for package in sorted(_TEST_SEMANTIC_PACKAGES):
        assert (tests_root / package).is_dir()
    assert sorted(path.name for path in tests_root.glob("*.py")) == sorted(_ALLOWED_TEST_ROOT_FILES)
    assert sorted(path.name for path in tests_root.iterdir() if path.is_dir() and path.name != "__pycache__") == sorted(
        _TEST_SEMANTIC_PACKAGES
    )
    assert not list(tests_root.glob("test_*.py"))


def test_masking_package_owns_shared_mask_algebra() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/masking"
    test_root = repo_root() / _PACKAGE_ROOT / "tests/masking"

    assert sorted(path.name for path in source_root.glob("*.py")) == sorted(_MASKING_ROOT_FILES)
    assert sorted(path.name for path in test_root.glob("*.py")) == sorted(_MASKING_TEST_ROOT_FILES)


def test_materialization_primitives_are_semantic_packages() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/materialization"
    test_root = repo_root() / _PACKAGE_ROOT / "tests/materialization"

    assert sorted(path.name for path in source_root.glob("*.py")) == ["__init__.py"]
    assert sorted(path.name for path in test_root.glob("*.py")) == ["__init__.py"]
    assert sorted(
        path.name for path in source_root.iterdir() if path.is_dir() and path.name != "__pycache__"
    ) == sorted(_MATERIALIZATION_PRIMITIVES)
    assert sorted(path.name for path in test_root.iterdir() if path.is_dir() and path.name != "__pycache__") == sorted(
        _MATERIALIZATION_PRIMITIVES
    )
    for primitive in sorted(_MATERIALIZATION_PRIMITIVES):
        assert (source_root / primitive / "__init__.py").is_file()
        assert (source_root / primitive / "__main__.py").is_file()
        assert (source_root / primitive / "pipeline.py").is_file()
        assert (test_root / primitive / "__init__.py").is_file()
        assert (test_root / primitive / "test_materialization.py").is_file()
