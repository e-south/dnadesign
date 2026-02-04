"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_require_non_null_cols.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.baserender.src.contracts import SchemaError
from dnadesign.baserender.src.io.parquet import read_parquet_records


def _write_parquet(path):
    seqs = ["ACGTTT", "GGGCCC"]
    anns = [
        [{"offset": 0, "orientation": "fwd", "tf": "lexa", "tfbs": "ACG"}],
        [{"offset": 0, "orientation": "fwd", "tf": "cpxr", "tfbs": "GGG"}],
    ]
    table = pa.table(
        {
            "sequence": seqs,
            "densegen__used_tfbs_detail": anns,
            "meta": ["ok", ""],
        }
    )
    pq.write_table(table, path)


def test_require_non_null_cols_missing_column_raises_schemaerror(tmp_path):
    path = tmp_path / "records.parquet"
    _write_parquet(path)
    with pytest.raises(SchemaError):
        _ = list(
            read_parquet_records(
                path,
                ann_policy={"require_non_null_cols": ["does_not_exist"]},
            )
        )


def test_require_non_null_cols_present_enforced(tmp_path):
    path = tmp_path / "records.parquet"
    _write_parquet(path)
    recs = list(
        read_parquet_records(
            path,
            ann_policy={"require_non_null_cols": ["meta"]},
        )
    )
    assert [r.id for r in recs] == ["row_0"]
