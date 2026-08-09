"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/tests/reader_records/test_layout.py

Keep Reader-record boundary tests organized by behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path


def test_reader_record_tests_have_a_bounded_semantic_layout() -> None:
    tests_dir = Path(__file__).parent
    assert {path.name for path in tests_dir.glob("test_*.py")} == {
        "test_identity.py",
        "test_layout.py",
        "test_pagination.py",
        "test_path_confinement.py",
        "test_verification.py",
    }
    assert not (tests_dir.parent / "test_reader_records.py").exists()
