"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/tests/test_tombstones.py

Tombstone overlay behavior tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dnadesign.usr import Dataset


def _row(seq: str, *, source: str = "test") -> dict:
    return {
        "sequence": seq,
        "bio_type": "dna",
        "alphabet": "dna_4",
        "source": source,
    }


def _make_dataset(root: Path) -> Dataset:
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")
    ds.import_rows([_row("ACGT"), _row("TGCA")], source="unit-test")
    return ds


def test_tombstone_excludes_by_default(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)

    ids = ds.head(2)["id"].tolist()
    ds.tombstone([ids[0]], reason="bad")

    df = ds.head(10)
    assert ids[0] not in df["id"].tolist()
    assert "usr__deleted" not in df.columns

    df_all = ds.head(10, include_deleted=True)
    assert ids[0] in df_all["id"].tolist()
    assert "usr__deleted" in df_all.columns


def test_tombstone_restore(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)

    ids = ds.head(2)["id"].tolist()
    ds.tombstone([ids[0]], reason="bad")
    ds.restore([ids[0]])

    df = ds.head(10)
    assert ids[0] in df["id"].tolist()


def test_export_include_deleted(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)

    ids = ds.head(2)["id"].tolist()
    ds.tombstone([ids[0]], reason="bad")

    out_default = tmp_path / "out_default.csv"
    ds.export("csv", out_default)
    df_default = pd.read_csv(out_default)
    assert ids[0] not in df_default["id"].tolist()

    out_all = tmp_path / "out_all.csv"
    ds.export("csv", out_all, include_deleted=True)
    df_all = pd.read_csv(out_all)
    assert ids[0] in df_all["id"].tolist()
