"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/tests/test_dataset_read_ops.py

Read-path tests for Dataset helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

import pandas as pd

from dnadesign.usr.src.dataset import Dataset


def _make_dataset(tmp_path: Path) -> Dataset:
    ds = Dataset(tmp_path, "ns/demo")
    ds.init(source="test")
    ds.import_rows(
        [
            {
                "sequence": "ACGT",
                "bio_type": "dna",
                "alphabet": "dna_4",
                "source": "unit",
            },
            {
                "sequence": "TGCA",
                "bio_type": "dna",
                "alphabet": "dna_4",
                "source": "unit",
            },
        ]
    )
    return ds


def test_head_columns(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    df = ds.head(1, columns=["id", "sequence"])
    assert list(df.columns) == ["id", "sequence"]
    assert len(df) == 1


def test_get_columns(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    full = ds.head(1)
    rid = full.iloc[0]["id"]
    got = ds.get(rid, columns=["id", "sequence"])
    assert isinstance(got, pd.DataFrame)
    assert list(got.columns) == ["id", "sequence"]
    assert got.iloc[0]["id"] == rid


def test_export_columns(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    out = tmp_path / "out.csv"
    ds.export("csv", out, columns=["id", "sequence"])
    data = pd.read_csv(out)
    assert list(data.columns) == ["id", "sequence"]


def test_grep_batch_size(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    df = ds.grep("AC", limit=1, batch_size=1)
    assert len(df) <= 1
