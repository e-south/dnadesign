"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/tests/test_locking.py

Tests dataset write lock behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

from dnadesign.usr import Dataset
from dnadesign.usr.src import dataset as dataset_module
from dnadesign.usr.src.locks import LOCK_FILENAME, dataset_write_lock
from dnadesign.usr.tests.registry_helpers import ensure_registry


def test_dataset_write_lock_creates_lock_file(tmp_path: Path) -> None:
    lock_path = tmp_path / LOCK_FILENAME
    with dataset_write_lock(tmp_path):
        assert lock_path.exists()


def test_import_rows_uses_write_lock(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "datasets"
    ensure_registry(root)
    ds = Dataset(root, "demo")
    ds.init(source="test")
    called = {"lock": False}

    def _fake_lock(_path):
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            called["lock"] = True
            yield

        return _ctx()

    monkeypatch.setattr(dataset_module, "dataset_write_lock", _fake_lock)
    ds.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "test"},
        ],
        source="test",
    )
    assert called["lock"] is True
