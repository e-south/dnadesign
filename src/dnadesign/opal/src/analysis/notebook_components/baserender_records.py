from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .baserender import NO_RENDERABLE_RECORDS_LABEL


def build_notebook_baserender_record_options(
    records_path: str | Path,
    contract: Mapping[str, Any],
    *,
    labels_df: Any | None = None,
    round_value: Any | None = None,
    limit: int = 500,
) -> list[str]:
    """Return renderable record ids for the active BaseRender contract."""

    if not bool(contract.get("available")):
        return [NO_RENDERABLE_RECORDS_LABEL]

    import polars as pl

    source_columns = _source_columns(contract)
    if not source_columns:
        return [NO_RENDERABLE_RECORDS_LABEL]
    id_column = _id_column(contract)
    if id_column not in source_columns:
        source_columns.append(id_column)
    scan = pl.scan_parquet(str(records_path)).select(source_columns)
    schema = scan.collect_schema()
    for expr in _contract_valid_filters(pl, contract, schema):
        scan = scan.filter(expr)
    label_ids = _label_ids_for_round(labels_df, round_value=round_value)
    if label_ids:
        scan = scan.filter(pl.col(id_column).cast(pl.Utf8).is_in(label_ids))
    records = (
        scan.select(pl.col(id_column).cast(pl.Utf8).alias("__record_id"))
        .drop_nulls()
        .unique(maintain_order=True)
        .limit(max(1, int(limit)))
        .collect()
    )
    options = records.get_column("__record_id").to_list() if "__record_id" in records.columns else []
    return [str(item) for item in options if str(item).strip()] or [NO_RENDERABLE_RECORDS_LABEL]


def load_notebook_baserender_record_row(
    records_path: str | Path,
    record_id: str,
    contract: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Load one contract-valid record row for notebook BaseRender display."""

    if not bool(contract.get("available")) or str(record_id) == NO_RENDERABLE_RECORDS_LABEL:
        return None

    import polars as pl

    id_column = _id_column(contract)
    source_columns = _source_columns(contract)
    if id_column not in source_columns:
        source_columns.append(id_column)
    scan = pl.scan_parquet(str(records_path)).select(source_columns)
    schema = scan.collect_schema()
    for expr in _contract_valid_filters(pl, contract, schema):
        scan = scan.filter(expr)
    row_df = scan.filter(pl.col(id_column).cast(pl.Utf8) == str(record_id)).limit(1).collect()
    if row_df.is_empty():
        return None
    return row_df.to_dicts()[0]


def build_notebook_baserender_label_rows(
    labels_df: Any | None,
    *,
    record_id: str,
    round_value: Any | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return compact observed-label rows for the selected rendered record."""

    if labels_df is None or str(record_id) == NO_RENDERABLE_RECORDS_LABEL:
        return []
    if not hasattr(labels_df, "columns") or not hasattr(labels_df, "is_empty"):
        return []
    if "id" not in labels_df.columns or labels_df.is_empty():
        return []
    try:
        import polars as pl

        filtered = labels_df.filter(pl.col("id").cast(pl.Utf8) == str(record_id))
        if round_value is not None and "observed_round" in filtered.columns:
            filtered = filtered.filter(pl.col("observed_round") == int(round_value))
        label_columns = [
            column
            for column in (
                "observed_round",
                "id",
                "y_space",
                "y_obs",
                "src",
                "label_src",
                "note",
                "ts",
            )
            if column in filtered.columns
        ]
        if not label_columns or filtered.is_empty():
            return []
        return filtered.select(label_columns).head(max(1, int(limit))).to_dicts()
    except Exception:
        return []


def _id_column(contract: Mapping[str, Any]) -> str:
    adapter_columns = contract.get("adapter_columns")
    if isinstance(adapter_columns, Mapping):
        return str(adapter_columns.get("id") or "id")
    return "id"


def _required_columns(contract: Mapping[str, Any]) -> list[str]:
    return [str(column) for column in contract.get("required_columns") or () if str(column)]


def _source_columns(contract: Mapping[str, Any]) -> list[str]:
    out = _required_columns(contract)
    adapter_columns = contract.get("adapter_columns")
    if isinstance(adapter_columns, Mapping):
        for value in adapter_columns.values():
            if isinstance(value, str) and value and value not in out:
                out.append(value)
    return out


def _contract_valid_filters(pl: Any, contract: Mapping[str, Any], schema: Mapping[str, Any]) -> list[Any]:
    filters = []
    for column in _required_columns(contract):
        filters.append(pl.col(column).is_not_null())
        if str(schema.get(column)).startswith("List"):
            filters.append(pl.col(column).list.len() > 0)
    return filters


def _label_ids_for_round(labels_df: Any | None, *, round_value: Any | None) -> list[str]:
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
