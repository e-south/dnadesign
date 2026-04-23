"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/overlay_projection.py

Project namespaced overlay columns from one dataset onto another by join key.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd
import pyarrow as pa

from ..dataset import Dataset
from ..errors import SchemaError


@dataclass(frozen=True)
class OverlayProjectionPreview:
    src_dataset: str
    dest_dataset: str
    namespace: str
    src_join: str
    dest_join: str
    source_columns: tuple[str, ...]
    dest_rows: int
    matched_rows: int
    missing_rows: int


def _normalize_join_value(value: object) -> str | None:
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    return text or None


def _normalize_join_series(series: pd.Series) -> pd.Series:
    return series.map(_normalize_join_value)


def _prefixed_projection_columns(
    namespace: str,
    columns: Sequence[str] | None,
    *,
    available: Sequence[str],
) -> tuple[str, ...]:
    if columns is None:
        selected = [name for name in available if name.startswith(f"{namespace}__")]
        return tuple(selected)
    prefixed: list[str] = []
    for column in columns:
        cleaned = str(column or "").strip()
        if not cleaned:
            continue
        if "__" not in cleaned:
            cleaned = f"{namespace}__{cleaned}"
        prefixed.append(cleaned)
    return tuple(dict.fromkeys(prefixed))


def _scan_projection_frame(dataset: Dataset, *, columns: Sequence[str]) -> pd.DataFrame:
    batches = list(dataset.scan(columns=list(columns), include_overlays=True, batch_size=65_536))
    if not batches:
        return pd.DataFrame(columns=list(columns))
    return pa.Table.from_batches(batches).to_pandas()


def _duplicate_join_sample(series: pd.Series) -> str:
    dupes = series[series.notna() & series.duplicated(keep=False)]
    if dupes.empty:
        return ""
    sample_values = []
    seen: set[str] = set()
    for value in dupes.tolist():
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        sample_values.append(text)
        if len(sample_values) >= 5:
            break
    return ", ".join(sample_values)


def _arrow_ready_values(series: pd.Series) -> list[object]:
    values: list[object] = []
    for value in series.tolist():
        if value is None or value is pd.NA:
            values.append(None)
            continue
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
            converted = value.tolist()
            if isinstance(converted, (list, tuple, dict)):
                values.append(list(converted) if isinstance(converted, tuple) else converted)
                continue
            value = converted
        if isinstance(value, (list, tuple, dict)):
            values.append(list(value) if isinstance(value, tuple) else value)
            continue
        try:
            if pd.isna(value):
                values.append(None)
                continue
        except TypeError:
            pass
        values.append(value)
    return values


def project_namespace_overlay(
    *,
    root,
    src_dataset_name: str,
    dest_dataset_name: str,
    namespace: str,
    src_join: str = "id",
    dest_join: str = "id",
    columns: Sequence[str] | None = None,
    overwrite: bool = True,
    allow_missing: bool = False,
    dry_run: bool = False,
) -> OverlayProjectionPreview:
    src_dataset = Dataset(root, src_dataset_name)
    dest_dataset = Dataset(root, dest_dataset_name)
    src_schema = src_dataset.schema()
    dest_schema = dest_dataset.schema()

    source_columns = _prefixed_projection_columns(
        namespace,
        columns,
        available=src_schema.names,
    )
    if not source_columns:
        raise SchemaError(f"Dataset '{src_dataset.name}' does not expose any columns under namespace '{namespace}'.")

    required_src = [src_join, *source_columns]
    missing_src = [name for name in required_src if name not in src_schema.names]
    if missing_src:
        raise SchemaError(f"Source dataset '{src_dataset.name}' is missing required columns: {', '.join(missing_src)}.")
    required_dest = ["id", dest_join]
    missing_dest = [name for name in required_dest if name not in dest_schema.names]
    if missing_dest:
        raise SchemaError(
            f"Destination dataset '{dest_dataset.name}' is missing required columns: {', '.join(missing_dest)}."
        )

    src_frame = _scan_projection_frame(src_dataset, columns=required_src)
    dest_frame = _scan_projection_frame(dest_dataset, columns=required_dest)
    dest_rows = int(len(dest_frame))

    if dest_frame["id"].duplicated().any():
        raise SchemaError(f"Destination dataset '{dest_dataset.name}' has duplicate id values.")

    src_frame = src_frame.copy()
    dest_frame = dest_frame.copy()
    dest_id_column = "id"
    if dest_join == "id":
        dest_id_column = "id"
    else:
        dest_id_column = "__dest_id"
        dest_frame = dest_frame.rename(columns={"id": dest_id_column})
    src_frame[src_join] = _normalize_join_series(src_frame[src_join])
    dest_frame[dest_join] = _normalize_join_series(dest_frame[dest_join])

    src_non_null = src_frame[src_join].notna()
    if src_frame.loc[src_non_null, src_join].duplicated().any():
        sample = _duplicate_join_sample(src_frame[src_join])
        raise SchemaError(
            f"Source dataset '{src_dataset.name}' has duplicate join values for '{src_join}'."
            + (f" Sample: {sample}." if sample else "")
        )

    merged = dest_frame.merge(
        src_frame[[src_join, *source_columns]],
        how="left",
        left_on=dest_join,
        right_on=src_join,
        indicator=True,
        suffixes=("", "__src"),
        validate="many_to_one",
    )
    missing_rows = int(((merged[dest_join].notna()) & (merged["_merge"] == "left_only")).sum())
    if missing_rows and not allow_missing:
        sample = merged.loc[
            (merged[dest_join].notna()) & (merged["_merge"] == "left_only"),
            dest_join,
        ].astype(str)
        preview = ", ".join(sample.head(5).tolist())
        raise SchemaError(
            f"Destination dataset '{dest_dataset.name}' has {missing_rows} row(s) with '{dest_join}' values "
            f"that do not resolve in source dataset '{src_dataset.name}' via '{src_join}'."
            + (f" Sample: {preview}." if preview else "")
        )

    projected = merged.loc[merged["_merge"] == "both", [dest_id_column, *source_columns]].copy()
    if dest_id_column != "id":
        projected = projected.rename(columns={dest_id_column: "id"})
    matched_rows = int(len(projected))

    if not dry_run and matched_rows:
        arrays = {"id": pa.array(projected["id"].tolist(), type=pa.string())}
        fields = [pa.field("id", pa.string())]
        for column in source_columns:
            field = src_schema.field(column)
            arrays[column] = pa.array(_arrow_ready_values(projected[column]), type=field.type)
            fields.append(pa.field(column, field.type))
        dest_dataset.write_overlay(
            namespace,
            pa.table(arrays, schema=pa.schema(fields)),
            key="id",
            overwrite=overwrite,
            allow_missing=False,
        )

    return OverlayProjectionPreview(
        src_dataset=src_dataset.name,
        dest_dataset=dest_dataset.name,
        namespace=namespace,
        src_join=src_join,
        dest_join=dest_join,
        source_columns=source_columns,
        dest_rows=dest_rows,
        matched_rows=matched_rows,
        missing_rows=missing_rows,
    )
