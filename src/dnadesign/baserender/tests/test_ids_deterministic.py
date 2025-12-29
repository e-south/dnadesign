"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_ids_deterministic.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.baserender.src.contracts import SchemaError
from dnadesign.baserender.src.io.parquet import read_parquet_records


def _write_no_id_parquet(path):
    seqs = ["ACGTTT", "GGGCCC"]
    anns = [
        [{"offset": 0, "orientation": "fwd", "tf": "lexa", "tfbs": "ACG"}],
        [{"offset": 0, "orientation": "fwd", "tf": "cpxr", "tfbs": "GGG"}],
    ]
    table = pa.table(
        {
            "sequence": seqs,
            "densegen__used_tfbs_detail": anns,
        }
    )
    pq.write_table(table, path)


def _write_null_id_parquet(path):
    seqs = ["ACGTTT", "GGGCCC"]
    anns = [
        [{"offset": 0, "orientation": "fwd", "tf": "lexa", "tfbs": "ACG"}],
        [{"offset": 0, "orientation": "fwd", "tf": "cpxr", "tfbs": "GGG"}],
    ]
    table = pa.table(
        {
            "id": pa.array(["rec_a", None], type=pa.string()),
            "sequence": seqs,
            "densegen__used_tfbs_detail": anns,
        }
    )
    pq.write_table(table, path)


def test_deterministic_ids_when_missing_id_column(tmp_path):
    path = tmp_path / "no_id.parquet"
    _write_no_id_parquet(path)
    ids_first = [r.id for r in read_parquet_records(path)]
    ids_second = [r.id for r in read_parquet_records(path)]
    assert ids_first == ["row_0", "row_1"]
    assert ids_first == ids_second


def test_null_ids_are_skipped(tmp_path):
    path = tmp_path / "null_id.parquet"
    _write_null_id_parquet(path)
    ids = [r.id for r in read_parquet_records(path, id_col="id")]
    assert ids == ["rec_a"]


def test_id_col_declared_missing_raises_schemaerror(tmp_path):
    path = tmp_path / "no_id.parquet"
    _write_no_id_parquet(path)
    with pytest.raises(SchemaError):
        list(read_parquet_records(path, id_col="id"))


def test_details_col_declared_missing_raises_schemaerror(tmp_path):
    path = tmp_path / "no_id.parquet"
    _write_no_id_parquet(path)
    with pytest.raises(SchemaError):
        list(read_parquet_records(path, details_col="details"))
