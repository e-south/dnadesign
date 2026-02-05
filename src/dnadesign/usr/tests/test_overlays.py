"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/tests/test_overlays.py

Tests overlay-first attach and materialize behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from dnadesign.usr import Dataset, SchemaError
from dnadesign.usr.tests.registry_helpers import register_test_namespace


def _make_dataset(tmp_path: Path) -> Dataset:
    root = tmp_path / "datasets"
    register_test_namespace(root, namespace="mock", columns_spec="mock__score:float64")
    ds = Dataset(root, "demo")
    ds.init(source="test")
    ds.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "test"},
            {"sequence": "GGGG", "bio_type": "dna", "alphabet": "dna_4", "source": "test"},
        ],
        source="test",
    )
    return ds


def test_attach_creates_overlay_and_head_includes_derived(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    base_id = ds.head(1)["id"].iloc[0]
    df = pd.DataFrame({"id": [base_id], "score": [3.5]})
    path = tmp_path / "attach.csv"
    df.to_csv(path, index=False)

    ds.attach(path, namespace="mock", key="id", key_col="id", columns=["score"])

    overlay_path = ds.dir / "_derived" / "mock.parquet"
    assert overlay_path.exists()

    base_cols = pq.ParquetFile(str(ds.records_path)).schema_arrow.names
    assert "mock__score" not in base_cols

    head = ds.head(1)
    assert "mock__score" in head.columns
    assert head["mock__score"].iloc[0] == 3.5


def test_materialize_folds_overlays(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    base_id = ds.head(1)["id"].iloc[0]
    df = pd.DataFrame({"id": [base_id], "score": [1.0]})
    path = tmp_path / "attach.csv"
    df.to_csv(path, index=False)

    ds.attach(path, namespace="mock", key="id", key_col="id", columns=["score"])
    with pytest.raises(SchemaError):
        ds.materialize()
    ds.materialize(keep_overlays=False, maintenance=True)

    base_cols = pq.ParquetFile(str(ds.records_path)).schema_arrow.names
    assert "mock__score" in base_cols
    derived_dir = ds.dir / "_derived"
    assert not any(derived_dir.glob("*.parquet"))
