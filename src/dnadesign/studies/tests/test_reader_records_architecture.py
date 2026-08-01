"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_reader_records_architecture.py

Information-architecture contracts for the shared Reader record resolver.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.core import reader_records


def test_reader_records_uses_one_semantic_package_without_a_parallel_module() -> None:
    studies_root = Path(__file__).resolve().parents[1]
    core_root = studies_root / "core"
    package_root = core_root / "reader_records"

    assert not (core_root / "reader_records.py").exists()
    assert {path.name for path in package_root.iterdir() if path.is_file()} == {
        "__init__.py",
        "artifacts.py",
        "contracts.py",
        "resolver.py",
        "transport.py",
        "validation.py",
    }


def test_reader_records_package_exports_only_the_supported_contract_surface() -> None:
    assert set(reader_records.__all__) == {
        "READER_CATALOG_SCHEMA_VERSION",
        "READER_CLI_SCHEMA",
        "READER_RECORD_SCHEMA_VERSION",
        "ReaderArtifactFile",
        "ReaderDataframeRecordError",
        "ReaderDataframeRecordRef",
        "ReaderRecordError",
        "ReaderRecordExpectation",
        "ReaderRecordSet",
        "ReaderResolvedRecord",
        "resolve_digest_verified_dataframe_record",
        "resolve_digest_verified_records",
    }
