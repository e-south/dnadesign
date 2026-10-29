"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/baserender_record_sources.py

Notebook component builders for BaseRender record sources OPAL analysis notebook components.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ._support import compact_identifier


def id_column(contract: Mapping[str, Any]) -> str:
    adapter_columns = contract.get("adapter_columns")
    if isinstance(adapter_columns, Mapping):
        return str(adapter_columns.get("id") or "id")
    return "id"


def source_columns(contract: Mapping[str, Any]) -> list[str]:
    out = required_columns(contract)
    adapter_columns = contract.get("adapter_columns")
    if isinstance(adapter_columns, Mapping):
        for value in adapter_columns.values():
            if isinstance(value, str) and value and value not in out:
                out.append(value)
    return out


def record_source_columns(contract: Mapping[str, Any]) -> list[str]:
    metadata_path = metadata_records_path(contract)
    annotation = annotation_column(contract)
    out = required_columns(contract)
    adapter_columns = contract.get("adapter_columns")
    if isinstance(adapter_columns, Mapping):
        for key, value in adapter_columns.items():
            if metadata_path is not None and str(value) == str(annotation) and key in {"annotations", "features"}:
                continue
            if isinstance(value, str) and value and value not in out:
                out.append(value)
    return out


def metadata_source_columns(contract: Mapping[str, Any]) -> list[str]:
    out = [str(column) for column in contract.get("metadata_required_columns") or () if str(column)]
    identifier = id_column(contract)
    annotation = annotation_column(contract)
    for column in (identifier, annotation):
        if column and column not in out:
            out.append(column)
    return out


def metadata_records_path(contract: Mapping[str, Any]) -> str | None:
    path = str(contract.get("metadata_records_path") or "").strip()
    return path or None


def annotation_column(contract: Mapping[str, Any]) -> str | None:
    adapter_columns = contract.get("adapter_columns")
    if not isinstance(adapter_columns, Mapping):
        return None
    for key in ("annotations", "features"):
        value = adapter_columns.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def annotation_count_expr(pl: Any, column: str, schema: Mapping[str, Any]) -> Any:
    dtype_text = str(schema.get(column))
    if dtype_text.startswith("List"):
        return pl.col(column).list.len().fill_null(0).cast(pl.Int64).alias("__annotation_count")
    return pl.when(pl.col(column).is_null()).then(0).otherwise(1).cast(pl.Int64).alias("__annotation_count")


def join_metadata_ids(pl: Any, scan: Any, metadata_path: str, *, id_column_name: str) -> Any:
    metadata_ids = (
        pl.scan_parquet(metadata_path)
        .select(pl.col(id_column_name).cast(pl.Utf8).alias(id_column_name))
        .drop_nulls()
        .unique()
    )
    return scan.with_columns(pl.col(id_column_name).cast(pl.Utf8).alias(id_column_name)).join(
        metadata_ids,
        on=id_column_name,
        how="inner",
    )


def join_metadata_rows(
    pl: Any,
    scan: Any,
    metadata_path: str,
    *,
    id_column_name: str,
    contract: Mapping[str, Any],
) -> Any:
    metadata_columns = metadata_source_columns(contract)
    metadata = pl.scan_parquet(metadata_path).select(metadata_columns)
    return scan.with_columns(pl.col(id_column_name).cast(pl.Utf8).alias(id_column_name)).join(
        metadata.with_columns(pl.col(id_column_name).cast(pl.Utf8).alias(id_column_name)),
        on=id_column_name,
        how="inner",
    )


def require_unique_record_ids(pl: Any, scan: Any, *, id_column_name: str) -> None:
    """Fail when a selected BaseRender identity resolves to more than one row."""

    duplicate_ids = (
        scan.select(pl.col(id_column_name).cast(pl.Utf8).alias(id_column_name))
        .drop_nulls()
        .group_by(id_column_name)
        .agg(pl.len().alias("__row_count"))
        .filter(pl.col("__row_count") > 1)
        .sort(id_column_name)
        .limit(10)
        .collect()
        .get_column(id_column_name)
        .to_list()
    )
    if duplicate_ids:
        raise ValueError(f"Found duplicate BaseRender record id rows: {duplicate_ids}.")


def contract_valid_filters(pl: Any, contract: Mapping[str, Any], schema: Mapping[str, Any]) -> list[Any]:
    filters = []
    policies = contract.get("adapter_policies")
    if isinstance(policies, Mapping):
        require_non_empty = bool(policies.get("require_non_empty")) or int(policies.get("min_per_record") or 0) > 0
    else:
        require_non_empty = False
    for column in required_columns(contract):
        filters.append(pl.col(column).is_not_null())
        if require_non_empty and str(schema.get(column)).startswith("List"):
            filters.append(pl.col(column).list.len() > 0)
    return filters


def label_ids_for_round(labels_df: Any | None, *, round_value: Any | None) -> list[str]:
    if labels_df is None or round_value is None:
        return []
    if not hasattr(labels_df, "columns") or not hasattr(labels_df, "is_empty"):
        return []
    if "id" not in labels_df.columns or "observed_round" not in labels_df.columns or labels_df.is_empty():
        return []
    try:
        import polars as pl

        return (
            labels_df.filter(pl.col("observed_round") == int(round_value))
            .select(pl.col("id").cast(pl.Utf8))
            .drop_nulls()
            .unique(maintain_order=True)
            .get_column("id")
            .to_list()
        )
    except Exception:
        return []


def normalise_record_ids(record_ids: Iterable[Any] | None) -> list[str]:
    if record_ids is None:
        return []
    values: list[str] = []
    seen: set[str] = set()
    for value in record_ids:
        text = str(value).strip()
        if text and text not in seen:
            values.append(text)
            seen.add(text)
    return values


def compact_record_id(record_id: str) -> str:
    return compact_identifier(record_id)


def required_columns(contract: Mapping[str, Any]) -> list[str]:
    return [str(column) for column in contract.get("required_columns") or () if str(column)]
