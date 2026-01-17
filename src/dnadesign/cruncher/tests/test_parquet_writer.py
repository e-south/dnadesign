"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_parquet_writer.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pyarrow.parquet as pq

from dnadesign.cruncher.app.sample_workflow import _elite_parquet_schema, _write_parquet_rows


def test_write_parquet_rows_creates_empty_elites(tmp_path) -> None:
    path = tmp_path / "elites.parquet"
    schema = _elite_parquet_schema(["tfA", "tfB"], include_canonical=False)
    count = _write_parquet_rows(path, iter([]), schema=schema)
    assert count == 0
    assert path.exists()
    table = pq.read_table(path)
    expected = {
        "sequence",
        "rank",
        "norm_sum",
        "min_norm",
        "sum_norm",
        "combined_score_final",
        "chain",
        "draw_idx",
        "meta_type",
        "meta_source",
        "meta_date",
        "per_tf_json",
        "score_tfA",
        "score_tfB",
        "norm_tfA",
        "norm_tfB",
    }
    assert expected.issubset(set(table.column_names))
