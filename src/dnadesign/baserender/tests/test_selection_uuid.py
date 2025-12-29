"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_selection_uuid.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import uuid

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.baserender.src.io.parquet import (
    canonicalize_id_strings_for_parquet,
    read_parquet_records_by_ids,
)


def _write_uuid_parquet(path, uuids):
    seqs = ["ACGTTT", "GGGCCC"]
    anns = [
        [{"offset": 0, "orientation": "fwd", "tf": "lexa", "tfbs": "ACG"}],
        [{"offset": 0, "orientation": "fwd", "tf": "cpxr", "tfbs": "GGG"}],
    ]
    fixed_bin = getattr(pa, "fixed_size_binary", None)
    id_type = fixed_bin(16) if fixed_bin is not None else pa.binary(16)
    table = pa.table(
        {
            "id": pa.array([u.bytes for u in uuids], type=id_type),
            "sequence": seqs,
            "densegen__used_tfbs_detail": anns,
        }
    )
    pq.write_table(table, path)


def test_selection_uuid_binary_canonicalizes(tmp_path):
    ids = [uuid.uuid4(), uuid.uuid4()]
    path = tmp_path / "records.parquet"
    _write_uuid_parquet(path, ids)

    raw_ids = [u.hex for u in ids]
    canonical_ids, _ = canonicalize_id_strings_for_parquet(path, id_col="id", raw_ids=raw_ids)
    assert canonical_ids == [str(u) for u in ids]

    recs = list(
        read_parquet_records_by_ids(
            path,
            ids=canonical_ids,
            sequence_col="sequence",
            annotations_col="densegen__used_tfbs_detail",
            id_col="id",
        )
    )
    assert {r.id for r in recs} == {str(u) for u in ids}
