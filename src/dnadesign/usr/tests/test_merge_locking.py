"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_merge_locking.py

Ensure merge-datasets uses dataset write lock.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from contextlib import contextmanager
from pathlib import Path

from dnadesign.usr import Dataset
from dnadesign.usr.src import merge_datasets as merge_module
from dnadesign.usr.src.merge_datasets import MergeColumnsMode, MergePolicy, merge_usr_to_usr
from dnadesign.usr.tests.registry_helpers import ensure_registry


def _row(seq: str, *, source: str = "test") -> dict:
    return {
        "sequence": seq,
        "bio_type": "dna",
        "alphabet": "dna_4",
        "source": source,
    }


def test_merge_uses_write_lock(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "datasets"
    ensure_registry(root)
    dest = Dataset(root, "dest")
    src = Dataset(root, "src")
    dest.init(source="unit-test")
    src.init(source="unit-test")
    dest.import_rows([_row("ACGT")], source="unit-test")
    src.import_rows([_row("TGCA")], source="unit-test")

    called = {"lock": False}

    def _lock(_path):
        @contextmanager
        def _ctx():
            called["lock"] = True
            yield

        return _ctx()

    monkeypatch.setattr(merge_module, "dataset_write_lock", _lock)

    with dest.maintenance(reason="merge"):
        merge_usr_to_usr(
            root=root,
            dest="dest",
            src="src",
            columns_mode=MergeColumnsMode.UNION,
            duplicate_policy=MergePolicy.SKIP,
            dry_run=False,
        )
    assert called["lock"] is True
