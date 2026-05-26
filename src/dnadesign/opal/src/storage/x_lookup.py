"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/storage/x_lookup.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..core.round_context import PluginCtx
from ..core.utils import OpalError
from ..registries.transforms_x import get_transform_x


def transform_rows_from_matching_row_groups(
    *,
    records_path: Path,
    id_list: Sequence[str],
    x_col: str,
    x_transform_name: str,
    x_transform_params: Mapping[str, Any],
    ctx: PluginCtx,
) -> dict[str, np.ndarray]:
    if ctx is None:
        raise OpalError("transform_matrix_from_records requires a PluginCtx for transform_x.")
    if not id_list:
        raise OpalError("transform_matrix_from_records requires at least one id.")
    if len(id_list) != len(set(id_list)):
        raise OpalError("transform_matrix_from_records received duplicate ids.")

    wanted = set(id_list)
    try:
        parquet = pq.ParquetFile(records_path)
    except Exception as exc:
        raise OpalError(f"Failed to read records.parquet for X lookup: {records_path}: {exc}") from exc

    schema_names = set(parquet.schema_arrow.names)
    missing_cols = [column for column in ("id", x_col) if column not in schema_names]
    if missing_cols:
        raise OpalError(f"records.parquet missing required column(s) for X lookup: {missing_cols}")

    tx = get_transform_x(x_transform_name, dict(x_transform_params))
    rows: dict[str, np.ndarray] = {}
    duplicates: set[str] = set()
    for row_group_index in range(parquet.num_row_groups):
        id_table = parquet.read_row_group(row_group_index, columns=["id"])
        id_values = [str(value) for value in id_table.column("id").to_pylist()]
        matches = [(row_index, row_id) for row_index, row_id in enumerate(id_values) if row_id in wanted]
        if not matches:
            continue

        seen_in_group: set[str] = set()
        for _row_index, row_id in matches:
            if row_id in rows or row_id in seen_in_group:
                duplicates.add(row_id)
            seen_in_group.add(row_id)

        data_table = parquet.read_row_group(row_group_index, columns=["id", x_col])
        filtered = data_table.take(pa.array([row_index for row_index, _row_id in matches], type=pa.int64()))
        frame = filtered.to_pandas()
        frame["id"] = frame["id"].astype(str)
        series = frame[x_col]
        null_mask = series.isna()
        if null_mask.any():
            bad_ids = frame.loc[null_mask, "id"].tolist()[:10]
            raise OpalError(f"X column '{x_col}' is null for ids (sample={bad_ids}).")

        X = tx(series, ctx=ctx)
        if X.shape[0] != len(frame):
            raise OpalError(f"transform_x[{x_transform_name}] returned {X.shape[0]} rows for {len(frame)} ids.")
        for row_index, row_id in enumerate(frame["id"].tolist()):
            rows[str(row_id)] = np.asarray(X[row_index], dtype=float)

    if duplicates:
        raise OpalError(f"records.parquet contains duplicate requested ids (sample={sorted(duplicates)[:10]}).")
    missing = [row_id for row_id in id_list if row_id not in rows]
    if missing:
        raise OpalError(f"Missing ids in records.parquet for transform_matrix (sample={missing[:10]}).")
    return rows
