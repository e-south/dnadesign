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
_CLI_MATERIALIZATION_PACKAGES = {
    "candidate_table",
    "contact_geometry",
    "contact_risk",
    "foldcheck_report",
    "foldcheck_request",
    "manual_mask_authority",
    "mask_set",
    "proteinmpnn_request",
    "proteinmpnn_sample_ingest",
    "structure_preprocessing",
    "thread_plan",
}
_CONTACT_GEOMETRY_ROOT_FILES = {
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
_MATERIALIZATION_PRIMITIVES = {
    "candidate_table",
    "contact",
    "contact_geometry",
    "contact_risk",
    "conservation",
    "conservation_alignments",
    "foldcheck_report",
    "foldcheck_request",
    "manual_mask_authority",
    "mask_set",
    "proteinmpnn_request",
    "proteinmpnn_sample_ingest",
    "source_sequences",
    "structure",
    "structure_preprocessing",
    "thread_plan",
}
_PROTEINMPNN_REQUEST_ROOT_FILES = {
    "__init__.py",
    "__main__.py",
    "cli.py",
    "constants.py",
    "models.py",
    "pipeline.py",
}
_FOLDCHECK_REQUEST_ROOT_FILES = {
    "__init__.py",
    "__main__.py",
    "cli.py",
    "constants.py",
    "models.py",
    "pipeline.py",
    "sequences.py",
}
_FOLDCHECK_REPORT_ROOT_FILES = {
    "__init__.py",
    "__main__.py",
    "cli.py",
    "constants.py",
    "pipeline.py",
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


def test_cli_materializers_keep_cli_parsing_out_of_pipelines() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/materialization"

    for package in sorted(_CLI_MATERIALIZATION_PACKAGES):
        package_root = source_root / package
        assert (package_root / "cli.py").is_file()
        pipeline_text = (package_root / "pipeline.py").read_text(encoding="utf-8")
        assert "argparse" not in pipeline_text
        assert "def main(" not in pipeline_text


def test_contact_geometry_materializer_uses_semantic_modules() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/materialization/contact_geometry"

    assert sorted(path.name for path in source_root.glob("*.py")) == sorted(_CONTACT_GEOMETRY_ROOT_FILES)
    assert "Bio.PDB" not in (source_root / "pipeline.py").read_text(encoding="utf-8")
    assert "pyarrow as pa" not in (source_root / "pipeline.py").read_text(encoding="utf-8")
    assert "np.stack" not in (source_root / "pipeline.py").read_text(encoding="utf-8")
    assert "write_geometry_profile" in (source_root / "writer.py").read_text(encoding="utf-8")
    assert "distance_matrix" in (source_root / "rows.py").read_text(encoding="utf-8")
    assert "MMCIFParser" in (source_root / "structure_io.py").read_text(encoding="utf-8")


def test_proteinmpnn_request_materializer_uses_semantic_modules() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/materialization/proteinmpnn_request"

    assert sorted(path.name for path in source_root.glob("*.py")) == sorted(_PROTEINMPNN_REQUEST_ROOT_FILES)
    pipeline_text = (source_root / "pipeline.py").read_text(encoding="utf-8")
    assert "pyarrow" not in pipeline_text
    assert "hashlib" not in pipeline_text
    assert "protein_mpnn_run.py" not in pipeline_text
    assert "dnadesign.thread.adapters.proteinmpnn" in pipeline_text
    assert "build_request_manifest" in pipeline_text
    assert "export_chain_backbone" in pipeline_text


def test_foldcheck_request_materializer_uses_semantic_modules() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/materialization/foldcheck_request"

    assert sorted(path.name for path in source_root.glob("*.py")) == sorted(_FOLDCHECK_REQUEST_ROOT_FILES)
    pipeline_text = (source_root / "pipeline.py").read_text(encoding="utf-8")
    assert "pyarrow" not in pipeline_text
    assert "re.compile" not in pipeline_text
    assert "dnadesign.thread.foldcheck" in pipeline_text
    assert "build_foldcheck_sequence_records" in pipeline_text


def test_foldcheck_report_materializer_uses_thread_colabfold_adapter() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/materialization/foldcheck_report"

    assert sorted(path.name for path in source_root.glob("*.py")) == sorted(_FOLDCHECK_REPORT_ROOT_FILES)
    pipeline_text = (source_root / "pipeline.py").read_text(encoding="utf-8")
    assert "pyarrow" not in pipeline_text
    assert "np." not in pipeline_text
    assert "dnadesign.thread.adapters.colabfold" in pipeline_text
    assert "dnadesign.thread.foldcheck" in pipeline_text
