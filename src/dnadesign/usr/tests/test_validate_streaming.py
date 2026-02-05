"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/tests/test_validate_streaming.py

Tests streaming validation without full-table reads.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.usr import Dataset, DuplicateIDError
from dnadesign.usr.src import dataset as dataset_module
from dnadesign.usr.src.normalize import compute_id
from dnadesign.usr.src.schema import ARROW_SCHEMA


def test_validate_detects_duplicates_without_full_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds = Dataset(tmp_path, "demo")
    ds.init(source="test")

    rid = compute_id("dna", "ACGT")
    df = pd.DataFrame(
        {
            "id": [rid, rid],
            "bio_type": ["dna", "dna"],
            "sequence": ["ACGT", "ACGT"],
            "alphabet": ["dna_4", "dna_4"],
            "length": [4, 4],
            "source": ["test", "test"],
            "created_at": [pd.Timestamp("2020-01-01", tz="UTC")] * 2,
        }
    )
    tbl = pa.Table.from_pandas(df, schema=ARROW_SCHEMA, preserve_index=False)
    pq.write_table(tbl, ds.records_path)

    def _fail_read(*_args, **_kwargs):
        raise AssertionError("read_parquet should not be used by validate")

    monkeypatch.setattr(dataset_module, "read_parquet", _fail_read)

    with pytest.raises(DuplicateIDError):
        ds.validate()
