"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_annotation_contract.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.baserender.src.contracts import SchemaError
from dnadesign.baserender.src.io.parquet import read_parquet_records


def test_parse_ann_requires_tf(tmp_path):
    path = tmp_path / "records.parquet"
    table = pa.table(
        {
            "sequence": ["ACGT"],
            "densegen__used_tfbs_detail": [[{"offset": 0, "orientation": "fwd", "tfbs": "ACG"}]],
        }
    )
    pq.write_table(table, path)
    with pytest.raises(SchemaError):
        _ = list(read_parquet_records(path))


def test_parse_ann_orientation_case_insensitive(tmp_path):
    path = tmp_path / "records.parquet"
    table = pa.table(
        {
            "sequence": ["ACGT", "ACGT"],
            "densegen__used_tfbs_detail": [
                [{"offset": 0, "orientation": "FWD", "tf": "lexa", "tfbs": "ACG"}],
                [{"offset": 1, "orientation": "ReV", "tf": "lexa", "tfbs": "ACG"}],
            ],
        }
    )
    pq.write_table(table, path)
    recs = list(read_parquet_records(path))
    assert recs[0].annotations[0].strand == "fwd"
    assert recs[1].annotations[0].strand == "rev"
