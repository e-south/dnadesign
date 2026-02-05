"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_merge_streaming.py

Tests streamed merge implementation avoids full table loads.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

from dnadesign.usr import Dataset
from dnadesign.usr.src import merge_datasets as merge_module
from dnadesign.usr.src.merge_datasets import MergeColumnsMode, MergePolicy, merge_usr_to_usr


def _row(seq: str, *, source: str = "test") -> dict:
    return {
        "sequence": seq,
        "bio_type": "dna",
        "alphabet": "dna_4",
        "source": source,
    }


def test_merge_does_not_call_read_parquet(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "datasets"
    dest = Dataset(root, "dest")
    src = Dataset(root, "src")
    dest.init(source="unit-test")
    src.init(source="unit-test")
    dest.import_rows([_row("ACGT")], source="unit-test")
    src.import_rows([_row("TGCA")], source="unit-test")

    def _boom(*_args, **_kwargs):
        raise AssertionError("read_parquet should not be called during merge")

    monkeypatch.setattr(merge_module.pq, "read_table", _boom)

    preview = merge_usr_to_usr(
        root=root,
        dest="dest",
        src="src",
        columns_mode=MergeColumnsMode.UNION,
        duplicate_policy=MergePolicy.SKIP,
        dry_run=True,
        maintenance=True,
    )
    assert preview.dest_rows_before == 1
    assert preview.src_rows == 1
