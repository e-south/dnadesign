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
from .resume_policy import resolve_resume_filter_chunk_size


def _dedupe_ids(ids: List[str]) -> List[str]:
    seen: set[str] = set()
    unique_ids: List[str] = []
    for row_id in ids:
        normalized = str(row_id)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_ids.append(normalized)
    return unique_ids


def _positions_by_id(ids: List[str]) -> Dict[str, List[int]]:
    positions: Dict[str, List[int]] = {}
    for index, row_id in enumerate(ids):
        positions.setdefault(str(row_id), []).append(index)
    return positions


def _read_subset_table(
    *,
    pq,
    path: Path,
    columns: List[str],
    ids: List[str],
    filter_chunk_size: int,
):
    unique_ids = _dedupe_ids(ids)
    if len(unique_ids) <= filter_chunk_size:
        return (pq.read_table(path, columns=columns, filters=[("id", "in", unique_ids)]),)

    tables = []
    for start in range(0, len(unique_ids), filter_chunk_size):
        id_chunk = unique_ids[start : start + filter_chunk_size]
        tables.append(pq.read_table(path, columns=columns, filters=[("id", "in", id_chunk)]))
    return tuple(tables)


def _merge_table_values(
    *,
    existing: Dict[str, List[object]],
    outputs: List,
    infer_cols: Dict[str, str],
    table,
    positions: Dict[str, List[int]],
    only_non_null: bool,
) -> None:
    table_columns = set(table.schema.names)
    table_ids = table.column("id").to_pylist()
    for output in outputs:
        column_name = infer_cols[output.id]
        if column_name not in table_columns:
            continue
        values = table.column(column_name).to_pylist()
        target = existing[output.id]
        for table_index, table_id in enumerate(table_ids):
            target_positions = positions.get(str(table_id))
            if not target_positions:
                continue
            value = values[table_index]
            if only_non_null and value is None:
                continue
            for row_index in target_positions:
                target[row_index] = value


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
    id_positions = _positions_by_id(ids)

    try:
        import pyarrow.parquet as pq

        filter_chunk_size = resolve_resume_filter_chunk_size()

        records_path = ds.records_path  # type: ignore[attr-defined]
        records_parquet = pq.ParquetFile(records_path)
        records_columns = set(records_parquet.schema_arrow.names)  # type: ignore[attr-defined]
        selected_columns = ["id"] + [name for name in infer_cols.values() if name in records_columns]
        if len(selected_columns) > 1:
            for records_table in _read_subset_table(
                pq=pq,
                path=Path(records_path),
                columns=selected_columns,
                ids=ids,
                filter_chunk_size=filter_chunk_size,
            ):
                _merge_table_values(
                    existing=existing,
                    outputs=outputs,
                    infer_cols=infer_cols,
                    table=records_table,
                    positions=id_positions,
                    only_non_null=False,
                )

        if hasattr(ds, "list_overlays"):
            overlays = ds.list_overlays()  # type: ignore[attr-defined]
            infer_overlay = next((overlay for overlay in overlays if getattr(overlay, "namespace", None) == "infer"), None)
            if infer_overlay is not None:
                overlay_path = Path(str(infer_overlay.path))
                overlay_parquet = pq.ParquetFile(str(overlay_path))
                overlay_columns = set(overlay_parquet.schema_arrow.names)
                selected_overlay_columns = ["id"] + [name for name in infer_cols.values() if name in overlay_columns]
                if len(selected_overlay_columns) > 1:
                    for overlay_table in _read_subset_table(
                        pq=pq,
                        path=overlay_path,
                        columns=selected_overlay_columns,
                        ids=ids,
                        filter_chunk_size=filter_chunk_size,
                    ):
                        _merge_table_values(
                            existing=existing,
                            outputs=outputs,
                            infer_cols=infer_cols,
                            table=overlay_table,
                            positions=id_positions,
                            only_non_null=True,
                        )
    except Exception as exc:
        raise WriteBackError(f"USR resume scan failed for records table {ds.records_path}: {exc}") from exc

    todo_idx: List[int] = []
    for row_index in range(total_rows):
        if any(existing[output.id][row_index] is None for output in outputs):
            todo_idx.append(row_index)
    return todo_idx, existing
