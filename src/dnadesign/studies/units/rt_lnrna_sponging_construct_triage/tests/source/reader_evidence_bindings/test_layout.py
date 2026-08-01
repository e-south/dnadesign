"""Keep Reader evidence-binding tests organized by contract."""

from pathlib import Path

import dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence.bindings as bindings


def test_reader_evidence_binding_tests_have_a_bounded_semantic_layout() -> None:
    tests_dir = Path(__file__).parent
    assert {path.name for path in tests_dir.glob("test_*.py")} == {
        "test_cli.py",
        "test_identity.py",
        "test_layout.py",
        "test_persistence.py",
        "test_source_closure.py",
    }
    assert not (tests_dir.parent / "test_reader_evidence_bindings.py").exists()


def test_bindings_package_has_an_explicit_bounded_layout() -> None:
    package_dir = Path(bindings.__file__).parent
    assert {path.name for path in package_dir.glob("*.py")} == {
        "__init__.py",
        "building.py",
        "contracts.py",
        "persistence.py",
        "projection.py",
        "validation.py",
    }
    assert not package_dir.with_suffix(".py").exists()
    assert set(bindings.__all__) == {
        "BiologicalReplicateIdentityScope",
        "READER_EVIDENCE_BINDING_SCHEMA_ID",
        "ReaderEvidenceBinding",
        "ReaderEvidenceBindingError",
        "ReaderEvidenceBindingSet",
        "build_reader_evidence_bindings",
        "load_reader_evidence_bindings_json",
        "materialize_reader_evidence_bindings_json",
    }
