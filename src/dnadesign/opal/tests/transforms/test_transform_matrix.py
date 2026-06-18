"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/transforms/test_transform_matrix.py

Regression tests for transform matrix OPAL transforms.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.opal.src.core.round_context import PluginRegistryView, RoundCtx
from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.registries.transforms_x import get_transform_x
from dnadesign.opal.src.storage import x_lookup
from dnadesign.opal.src.storage.data_access import RecordsStore
from dnadesign.opal.src.transforms_x import identity  # noqa: F401 (registers identity)


def _store(tmp_path):
    return RecordsStore(
        kind="local",
        records_path=tmp_path / "records.parquet",
        campaign_slug="demo",
        x_col="X",
        y_col="Y",
        x_transform_name="identity",
        x_transform_params={},
    )


def _identity_ctx(tx):
    reg = PluginRegistryView("model", "objective", "selection", "identity", "transform_y")
    rctx = RoundCtx(core={"core/round_index": 0}, registry=reg)
    return rctx.for_plugin(category="transform_x", name="identity", plugin=tx)


def test_transform_matrix_preserves_order(tmp_path):
    df = pd.DataFrame(
        {
            "id": ["b", "a", "c"],
            "bio_type": ["dna", "dna", "dna"],
            "sequence": ["BBB", "AAA", "CCC"],
            "alphabet": ["dna_4", "dna_4", "dna_4"],
            "X": [[2.0], [1.0], [3.0]],
        }
    )
    store = _store(tmp_path)
    tx = get_transform_x("identity", {})
    tctx = _identity_ctx(tx)
    X, order = store.transform_matrix(df, ["b", "a"], ctx=tctx)
    assert order == ["b", "a"]
    assert X.tolist() == [[2.0], [1.0]]


def test_transform_matrix_from_records_reads_only_matching_x_row_groups(tmp_path, monkeypatch):
    records_path = tmp_path / "records.parquet"
    table = pa.Table.from_pandas(
        pd.DataFrame(
            {
                "id": ["a", "b", "c", "d"],
                "bio_type": ["dna", "dna", "dna", "dna"],
                "sequence": ["AAA", "BBB", "CCC", "DDD"],
                "alphabet": ["dna_4", "dna_4", "dna_4", "dna_4"],
                "X": [[1.0], [2.0], [3.0], [4.0]],
            }
        ),
        preserve_index=False,
    )
    pq.write_table(table, records_path, row_group_size=1)
    store = _store(tmp_path)
    tx = get_transform_x("identity", {})
    tctx = _identity_ctx(tx)

    real_parquet_file = x_lookup.pq.ParquetFile

    class CountingParquetFile:
        instances = []

        def __init__(self, *args, **kwargs):
            self._inner = real_parquet_file(*args, **kwargs)
            self.read_row_group_columns = []
            self.instances.append(self)

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def read_row_group(self, row_group_index, columns=None, *args, **kwargs):
            self.read_row_group_columns.append(tuple(columns or ()))
            return self._inner.read_row_group(row_group_index, columns=columns, *args, **kwargs)

    monkeypatch.setattr(x_lookup.pq, "ParquetFile", CountingParquetFile)

    X, order = store.transform_matrix_from_records(["d", "b"], ctx=tctx)

    assert order == ["d", "b"]
    assert X.tolist() == [[4.0], [2.0]]
    parquet_file = CountingParquetFile.instances[0]
    x_reads = [columns for columns in parquet_file.read_row_group_columns if columns == ("id", "X")]
    assert len(x_reads) == 2


def test_transform_matrix_from_records_rejects_duplicate_requested_record_ids(tmp_path):
    records_path = tmp_path / "records.parquet"
    table = pa.Table.from_pandas(
        pd.DataFrame(
            {
                "id": ["a", "a"],
                "bio_type": ["dna", "dna"],
                "sequence": ["AAA", "AAA2"],
                "alphabet": ["dna_4", "dna_4"],
                "X": [[1.0], [2.0]],
            }
        ),
        preserve_index=False,
    )
    pq.write_table(table, records_path, row_group_size=2)
    store = _store(tmp_path)
    tx = get_transform_x("identity", {})
    tctx = _identity_ctx(tx)

    with pytest.raises(OpalError, match="duplicate requested ids"):
        store.transform_matrix_from_records(["a"], ctx=tctx)


def test_iter_transform_matrix_batches_coalesces_sparse_matches(tmp_path):
    records_path = tmp_path / "records.parquet"
    table = pa.Table.from_pandas(
        pd.DataFrame(
            {
                "id": ["a", "b", "c", "d", "e", "f"],
                "bio_type": ["dna"] * 6,
                "sequence": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
                "alphabet": ["dna_4"] * 6,
                "X": [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]],
            }
        ),
        preserve_index=False,
    )
    pq.write_table(table, records_path, row_group_size=1)
    store = _store(tmp_path)
    tx = get_transform_x("identity", {})
    tctx = _identity_ctx(tx)

    batches = list(store.iter_transform_matrix_batches(["a", "c", "e"], ctx=tctx, batch_size=2))

    assert [ids for _X, ids in batches] == [["a", "c"], ["e"]]
    assert [X.tolist() for X, _ids in batches] == [[[1.0], [3.0]], [[5.0]]]


def test_iter_transform_matrix_batches_never_exceeds_batch_size_after_coalescing(tmp_path):
    records_path = tmp_path / "records.parquet"
    table = pa.Table.from_pandas(
        pd.DataFrame(
            {
                "id": ["a", "b", "c", "d", "e", "f"],
                "bio_type": ["dna"] * 6,
                "sequence": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
                "alphabet": ["dna_4"] * 6,
                "X": [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]],
            }
        ),
        preserve_index=False,
    )
    pq.write_table(table, records_path, row_group_size=3)
    store = _store(tmp_path)
    tx = get_transform_x("identity", {})
    tctx = _identity_ctx(tx)

    batches = list(store.iter_transform_matrix_batches(["a", "b", "d", "e"], ctx=tctx, batch_size=3))

    assert [ids for _X, ids in batches] == [["a", "b", "d"], ["e"]]
    assert all(X.shape[0] <= 3 for X, _ids in batches)


def test_transform_x_registry_requires_explicit_params() -> None:
    with pytest.raises(ValueError, match="params must be an explicit mapping"):
        get_transform_x("identity")


def test_identity_transform_rejects_scalar_runtime_cells() -> None:
    tx = get_transform_x("identity", {})

    with pytest.raises(ValueError, match="requires vector cells"):
        tx(pd.Series([1.0, 2.0]), ctx=_identity_ctx(tx))


def test_identity_transform_rejects_json_string_runtime_cells() -> None:
    tx = get_transform_x("identity", {})

    with pytest.raises(ValueError, match="requires vector cells"):
        tx(pd.Series(["[1.0, 2.0]", "[3.0, 4.0]"]), ctx=_identity_ctx(tx))


def test_transform_matrix_rejects_duplicate_ids(tmp_path):
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "bio_type": ["dna", "dna"],
            "sequence": ["AAA", "BBB"],
            "alphabet": ["dna_4", "dna_4"],
            "X": [[1.0], [2.0]],
        }
    )
    store = _store(tmp_path)
    with pytest.raises(OpalError):
        tx = get_transform_x("identity", {})
        tctx = _identity_ctx(tx)
        store.transform_matrix(df, ["a", "a"], ctx=tctx)
