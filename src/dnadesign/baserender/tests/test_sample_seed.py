"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_sample_seed.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.baserender.src.cli import _sample_records_by_row_index
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


def test_sample_seed_selects_by_row_index_under_gating(tmp_path):
    path = tmp_path / "records.parquet"
    _write_parquet_with_blank(path)
    recs = list(read_parquet_records(path))
    selected, idxs = _sample_records_by_row_index(recs, total_rows=3, k=1, seed=0)
    assert idxs == [1]
    # Row index 1 was dropped; selection should yield no records.
    assert selected == []
