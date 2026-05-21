"""
Focused tests for OPAL X-column validation contracts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.storage.records_io import RecordsIO
from dnadesign.opal.src.storage.x_contracts import validate_x_parquet_column


def _write_fixed_size_x(path: Path, values: list[list[float] | None]) -> None:
    table = pa.table(
        {
            "id": pa.array([f"id_{index}" for index in range(len(values))], type=pa.string()),
            "X": pa.array(values, type=pa.list_(pa.float32(), list_size=2)),
        }
    )
    pq.write_table(table, path)


def test_validate_x_parquet_column_accepts_fixed_size_list_without_pandas_materialization(tmp_path: Path) -> None:
    records = tmp_path / "records.parquet"
    _write_fixed_size_x(records, [[0.1, 0.2], [0.3, 0.4]])

    report = validate_x_parquet_column(records, x_column="X", batch_size=1)

    assert report.row_count == 2
    assert report.x_dim == 2


def test_validate_x_parquet_column_rejects_null_fixed_size_rows(tmp_path: Path) -> None:
    records = tmp_path / "records.parquet"
    _write_fixed_size_x(records, [[0.1, 0.2], None])

    with pytest.raises(OpalError, match="null or ragged fixed-size-list rows"):
        validate_x_parquet_column(records, x_column="X", batch_size=1)


def test_validate_x_parquet_column_rejects_nonfinite_fixed_size_values(tmp_path: Path) -> None:
    records = tmp_path / "records.parquet"
    _write_fixed_size_x(records, [[0.1, float("nan")], [0.3, 0.4]])

    with pytest.raises(OpalError, match="contains non-finite values for id 'id_0'"):
        validate_x_parquet_column(records, x_column="X", batch_size=1)


def test_validate_x_parquet_column_rejects_missing_columns(tmp_path: Path) -> None:
    records = tmp_path / "records.parquet"
    pq.write_table(pa.table({"id": pa.array(["a"], type=pa.string())}), records)

    with pytest.raises(OpalError, match="missing required column"):
        validate_x_parquet_column(records, x_column="X")


def test_validate_x_parquet_column_rejects_variable_list_physical_schema(tmp_path: Path) -> None:
    records = tmp_path / "records.parquet"
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["a", "b"], type=pa.string()),
                "X": pa.array([[0.1, 0.2], [0.3, 0.4]], type=pa.list_(pa.float32())),
            }
        ),
        records,
    )

    with pytest.raises(OpalError, match="must be stored as a Parquet fixed_size_list"):
        validate_x_parquet_column(records, x_column="X")


def test_validate_x_parquet_column_rejects_scalar_physical_schema(tmp_path: Path) -> None:
    records = tmp_path / "records.parquet"
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["a", "b"], type=pa.string()),
                "X": pa.array([0.1, 0.2], type=pa.float32()),
            }
        ),
        records,
    )

    with pytest.raises(OpalError, match="must be stored as a Parquet fixed_size_list"):
        validate_x_parquet_column(records, x_column="X")


def test_validate_x_parquet_column_rejects_integer_child_type(tmp_path: Path) -> None:
    records = tmp_path / "records.parquet"
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["a", "b"], type=pa.string()),
                "X": pa.array([[1, 2], [3, 4]], type=pa.list_(pa.int32(), list_size=2)),
            }
        ),
        records,
    )

    with pytest.raises(OpalError, match="float32 or float64"):
        validate_x_parquet_column(records, x_column="X")


def test_records_save_atomic_preserves_fixed_size_x_schema(tmp_path: Path) -> None:
    records = tmp_path / "records.parquet"
    _write_fixed_size_x(records, [[0.1, 0.2], [0.3, 0.4]])
    df = pd.read_parquet(records, engine="pyarrow")
    df["Y"] = pd.Series([[0.1], None], dtype=object)

    RecordsIO(records).save_atomic(df)

    report = validate_x_parquet_column(records, x_column="X")
    assert report.x_dim == 2
