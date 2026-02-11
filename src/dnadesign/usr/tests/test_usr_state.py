"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/tests/test_usr_state.py

Tests for usr_state convenience APIs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from dnadesign.usr import Dataset
from dnadesign.usr.src.errors import SchemaError
from dnadesign.usr.src.overlays import overlay_path
from dnadesign.usr.tests.registry_helpers import register_test_namespace


def _make_dataset(root: Path) -> Dataset:
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")
    ds.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
            {"sequence": "TGCA", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
        ],
        source="unit",
    )
    return ds


def test_set_and_clear_state(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    register_test_namespace(root, namespace="mock", columns_spec="mock__score:float64")
    ds = _make_dataset(root)

    ids = ds.head(2)["id"].tolist()
    rows = ds.set_state(ids, masked=True, qc_status="pass", split="train")
    assert rows == 2

    tbl = pq.read_table(overlay_path(ds.dir, "usr_state"))
    data = {rid: idx for idx, rid in enumerate(tbl.column("id").to_pylist())}
    for rid in ids:
        idx = data[rid]
        assert tbl.column("usr_state__masked")[idx].as_py() is True
        assert tbl.column("usr_state__qc_status")[idx].as_py() == "pass"
        assert tbl.column("usr_state__split")[idx].as_py() == "train"

    df = ds.get_state(ids)
    assert df.shape[0] == 2
    assert df["usr_state__masked"].tolist() == [True, True]

    rows = ds.clear_state(ids)
    assert rows == 2
    tbl = pq.read_table(overlay_path(ds.dir, "usr_state"))
    data = {rid: idx for idx, rid in enumerate(tbl.column("id").to_pylist())}
    for rid in ids:
        idx = data[rid]
        assert tbl.column("usr_state__masked")[idx].as_py() is False
        assert tbl.column("usr_state__qc_status")[idx].as_py() is None
        assert tbl.column("usr_state__split")[idx].as_py() is None


def test_set_state_rejects_invalid_qc_status(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    register_test_namespace(root, namespace="mock", columns_spec="mock__score:float64")
    ds = _make_dataset(root)
    ids = ds.head(1)["id"].tolist()
    with pytest.raises(SchemaError, match="usr_state__qc_status"):
        ds.set_state(ids, qc_status="maybe")


def test_set_state_rejects_invalid_split(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    register_test_namespace(root, namespace="mock", columns_spec="mock__score:float64")
    ds = _make_dataset(root)
    ids = ds.head(1)["id"].tolist()
    with pytest.raises(SchemaError, match="usr_state__split"):
        ds.set_state(ids, split="dev")
