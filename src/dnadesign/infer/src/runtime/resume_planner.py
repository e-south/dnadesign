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

from dnadesign.usr.src.overlays import overlay_metadata, overlay_parts

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


def _ordered_overlay_part_paths(path: Path) -> List[Path]:
    resolved = Path(str(path))
    if not resolved.exists():
        return []
    if resolved.is_file():
        return [resolved]

    parts = [Path(part) for part in overlay_parts(resolved)]
    if len(parts) <= 1:
        return parts

    return sorted(
        parts,
        key=lambda part: (
            str(overlay_metadata(part).get("created_at") or ""),
            part.name,
        ),
    )


def _infer_overlay_paths(ds) -> List[Path]:
    if not hasattr(ds, "list_overlays"):
        return []
    overlays = ds.list_overlays()  # type: ignore[attr-defined]
    infer_overlay = next(
        (overlay for overlay in overlays if getattr(overlay, "namespace", None) == "infer"),
        None,
    )
    if infer_overlay is None:
        return []
    return _ordered_overlay_part_paths(Path(str(infer_overlay.path)))


def _iter_overlay_subset_tables(
    *,
    pq,
    overlay_paths: List[Path],
    columns: List[str],
    ids: List[str],
    filter_chunk_size: int,
):
    for overlay_path in overlay_paths:
        overlay_parquet = pq.ParquetFile(str(overlay_path))
        overlay_columns = set(overlay_parquet.schema_arrow.names)
        selected_overlay_columns = ["id"] + [name for name in columns if name in overlay_columns]
        if len(selected_overlay_columns) <= 1:
            continue
        yield from _read_subset_table(
            pq=pq,
            path=overlay_path,
            columns=selected_overlay_columns,
            ids=ids,
            filter_chunk_size=filter_chunk_size,
        )


def _merge_named_column_values(
    *,
    values_by_column: Dict[str, List[object]],
    table,
    positions: Dict[str, List[int]],
    only_non_null: bool,
) -> None:
    table_columns = set(table.schema.names)
    table_ids = table.column("id").to_pylist()
    for column_name, target in values_by_column.items():
        if column_name not in table_columns:
            continue
        values = table.column(column_name).to_pylist()
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

    infer_cols = {o.id: infer_usr_column_name(model_id=model_id, job_id=job_id, out_id=o.id) for o in outputs}
    column_targets = {infer_cols[output.id]: existing[output.id] for output in outputs}
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
                _merge_named_column_values(
                    values_by_column=column_targets,
                    table=records_table,
                    positions=id_positions,
                    only_non_null=False,
                )

        for overlay_table in _iter_overlay_subset_tables(
            pq=pq,
            overlay_paths=_infer_overlay_paths(ds),
            columns=list(infer_cols.values()),
            ids=ids,
            filter_chunk_size=filter_chunk_size,
        ):
            _merge_named_column_values(
                values_by_column=column_targets,
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


def read_usr_columns(
    *,
    ds,
    ids: List[str],
    column_names: List[str],
) -> Dict[str, List[object]]:
    ordered_column_names = list(dict.fromkeys(str(name) for name in column_names if str(name).strip()))
    values: Dict[str, List[object]] = {column_name: [None] * len(ids) for column_name in ordered_column_names}
    if ds is None or len(ids) == 0 or len(ordered_column_names) == 0:
        return values

    id_positions = _positions_by_id(ids)

    try:
        import pyarrow.parquet as pq

        filter_chunk_size = resolve_resume_filter_chunk_size()

        records_path = ds.records_path  # type: ignore[attr-defined]
        records_parquet = pq.ParquetFile(records_path)
        records_columns = set(records_parquet.schema_arrow.names)  # type: ignore[attr-defined]
        selected_columns = ["id"] + [name for name in ordered_column_names if name in records_columns]
        if len(selected_columns) > 1:
            for records_table in _read_subset_table(
                pq=pq,
                path=Path(records_path),
                columns=selected_columns,
                ids=ids,
                filter_chunk_size=filter_chunk_size,
            ):
                _merge_named_column_values(
                    values_by_column=values,
                    table=records_table,
                    positions=id_positions,
                    only_non_null=False,
                )

        for overlay_table in _iter_overlay_subset_tables(
            pq=pq,
            overlay_paths=_infer_overlay_paths(ds),
            columns=ordered_column_names,
            ids=ids,
            filter_chunk_size=filter_chunk_size,
        ):
            _merge_named_column_values(
                values_by_column=values,
                table=overlay_table,
                positions=id_positions,
                only_non_null=True,
            )
    except Exception as exc:
        raise WriteBackError(f"USR column scan failed for records table {ds.records_path}: {exc}") from exc

    return values


def read_usr_column_values(
    *,
    ds,
    ids: List[str],
    column_name: str,
) -> List[object]:
    return read_usr_columns(ds=ds, ids=ids, column_names=[column_name]).get(column_name, [None] * len(ids))
