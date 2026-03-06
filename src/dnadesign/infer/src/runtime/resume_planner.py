"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/runtime/resume_planner.py

Plans resumable USR extract work by reading records and infer overlays.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from ..contracts import infer_usr_column_name
from ..errors import WriteBackError


def plan_resume_for_usr(
    *,
    ds,  # dnadesign.usr.Dataset
    ids: List[str],
    model_id: str,
    job_id: str,
    outputs: List,  # list[OutputSpec]
    overwrite: bool,
) -> Tuple[List[int], Dict[str, List[object]]]:
    total_rows = len(ids)
    existing: Dict[str, List[object]] = {o.id: [None] * total_rows for o in outputs}
    if overwrite or ds is None or total_rows == 0:
        return list(range(total_rows)), existing

    infer_cols = {
        o.id: infer_usr_column_name(model_id=model_id, job_id=job_id, out_id=o.id)
        for o in outputs
    }

    try:
        import pyarrow.parquet as pq

        records_path = ds.records_path  # type: ignore[attr-defined]
        records_parquet = pq.ParquetFile(records_path)
        records_columns = set(records_parquet.schema_arrow.names)  # type: ignore[attr-defined]
        selected_columns = ["id"] + [name for name in infer_cols.values() if name in records_columns]
        if len(selected_columns) > 1:
            records_table = pq.read_table(records_path, columns=selected_columns)
            record_ids = records_table.column("id").to_pylist()
            record_index = {record_id: index for index, record_id in enumerate(record_ids)}

            for output in outputs:
                column_name = infer_cols[output.id]
                if column_name in records_columns:
                    values = records_table.column(column_name).to_pylist()
                    for row_index, row_id in enumerate(ids):
                        table_index = record_index.get(row_id)
                        if table_index is not None:
                            existing[output.id][row_index] = values[table_index]

        if hasattr(ds, "list_overlays"):
            overlays = ds.list_overlays()  # type: ignore[attr-defined]
            infer_overlay = next((overlay for overlay in overlays if getattr(overlay, "namespace", None) == "infer"), None)
            if infer_overlay is not None:
                overlay_path = Path(str(infer_overlay.path))
                overlay_parquet = pq.ParquetFile(str(overlay_path))
                overlay_columns = set(overlay_parquet.schema_arrow.names)
                selected_overlay_columns = ["id"] + [name for name in infer_cols.values() if name in overlay_columns]
                if len(selected_overlay_columns) > 1:
                    overlay_table = pq.read_table(str(overlay_path), columns=selected_overlay_columns)
                    overlay_ids = overlay_table.column("id").to_pylist()
                    overlay_index = {record_id: index for index, record_id in enumerate(overlay_ids)}
                    for output in outputs:
                        column_name = infer_cols[output.id]
                        if column_name not in overlay_columns:
                            continue
                        values = overlay_table.column(column_name).to_pylist()
                        for row_index, row_id in enumerate(ids):
                            table_index = overlay_index.get(row_id)
                            if table_index is not None and values[table_index] is not None:
                                existing[output.id][row_index] = values[table_index]
    except Exception as exc:
        raise WriteBackError(f"USR resume scan failed for records table {ds.records_path}: {exc}") from exc

    todo_idx: List[int] = []
    for row_index in range(total_rows):
        if any(existing[output.id][row_index] is None for output in outputs):
            todo_idx.append(row_index)
    return todo_idx, existing
