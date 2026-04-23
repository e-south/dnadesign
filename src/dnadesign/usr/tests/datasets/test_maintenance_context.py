"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/tests/test_maintenance_context.py

Maintenance context enforcement tests for USR operations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.usr.src.dataset import Dataset
from dnadesign.usr.src.errors import SchemaError
from dnadesign.usr.tests.registry_helpers import register_test_namespace


def _make_dataset(root: Path) -> Dataset:
    register_test_namespace(root, namespace="audit", columns_spec="audit__score:float64")
    ds = Dataset(root, "demo")
    ds.init(source="unit")
    ds.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
            {"sequence": "TGCA", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
        ],
        source="unit",
    )
    return ds


def test_materialize_requires_maintenance_context(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)

    attach_path = tmp_path / "attach.parquet"
    ids = ds.head(2)["id"].tolist()
    pq.write_table(pa.table({"id": ids, "score": [0.1, 0.2]}), attach_path)
    ds.attach(attach_path, namespace="audit", key="id", backend="pyarrow", parse_json=False)

    with pytest.raises(SchemaError):
        ds.materialize()


def test_materialize_allows_maintenance_context(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)

    attach_path = tmp_path / "attach.parquet"
    ids = ds.head(2)["id"].tolist()
    pq.write_table(pa.table({"id": ids, "score": [0.1, 0.2]}), attach_path)
    ds.attach(attach_path, namespace="audit", key="id", backend="pyarrow", parse_json=False)

    with ds.maintenance(reason="materialize"):
        ds.materialize()

    schema = pq.ParquetFile(str(ds.records_path)).schema_arrow
    assert "audit__score" in schema.names
