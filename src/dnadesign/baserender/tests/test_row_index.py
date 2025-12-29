"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_row_index.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.baserender.src.io.parquet import read_parquet_records


def _write_parquet_with_blank(path):
    seqs = ["ACGTTT", "", "GGGCCC"]
    anns = [
        [{"offset": 0, "orientation": "fwd", "tf": "lexa", "tfbs": "ACG"}],
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


def test_row_index_is_parquet_row_even_when_rows_skipped(tmp_path):
    path = tmp_path / "records.parquet"
    _write_parquet_with_blank(path)
    recs = list(read_parquet_records(path))
    assert [r.row_index for r in recs] == [0, 2]
