"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_selection_row.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.baserender.src.cli import _select_records_by_row_index
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


def test_selection_match_on_row_uses_row_index_not_stream_index(tmp_path):
    path = tmp_path / "records.parquet"
    _write_parquet_with_blank(path)
    recs = list(read_parquet_records(path))
    found = _select_records_by_row_index(recs, [2])
    assert list(found.keys()) == [2]
    assert found[2].row_index == 2
