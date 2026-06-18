"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/baserender_records.py

Notebook component builders for BaseRender records OPAL analysis notebook components.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from .baserender import NO_RENDERABLE_RECORDS_LABEL
from .baserender_record_sources import (
    annotation_column,
    annotation_count_expr,
    compact_record_id,
    contract_valid_filters,
    id_column,
    join_metadata_ids,
    join_metadata_rows,
    label_ids_for_round,
    metadata_records_path,
    metadata_source_columns,
    normalise_record_ids,
    record_source_columns,
)
from .baserender_record_sources import (
    source_columns as contract_source_columns,
)


def build_notebook_baserender_record_options(
    records_path: str | Path,
    contract: Mapping[str, Any],
    *,
    labels_df: Any | None = None,
    round_value: Any | None = None,
    record_ids: Iterable[Any] | None = None,
    limit: int = 500,
) -> list[str]:
    """Return renderable record ids for the active BaseRender contract."""

    if not bool(contract.get("available")):
        return [NO_RENDERABLE_RECORDS_LABEL]

    import polars as pl

    source_columns = record_source_columns(contract)
    if not source_columns:
        return [NO_RENDERABLE_RECORDS_LABEL]
    identifier = id_column(contract)
    if identifier not in source_columns:
        source_columns.append(identifier)
    scan = pl.scan_parquet(str(records_path)).select(source_columns)
    schema = scan.collect_schema()
    for expr in contract_valid_filters(pl, contract, schema):
        scan = scan.filter(expr)
    metadata_path = metadata_records_path(contract)
    if metadata_path is not None:
        scan = join_metadata_ids(pl, scan, metadata_path, id_column_name=identifier)
    selected_ids = normalise_record_ids(record_ids)
    if record_ids is not None:
        if not selected_ids:
            return [NO_RENDERABLE_RECORDS_LABEL]
        scan = scan.filter(pl.col(identifier).cast(pl.Utf8).is_in(selected_ids))
    label_ids = label_ids_for_round(labels_df, round_value=round_value)
    if label_ids:
        scan = scan.filter(pl.col(identifier).cast(pl.Utf8).is_in(label_ids))
    records = (
        scan.select(pl.col(identifier).cast(pl.Utf8).alias("__record_id"))
        .drop_nulls()
        .unique(maintain_order=True)
        .limit(max(1, int(limit)))
        .collect()
    )
    options = records.get_column("__record_id").to_list() if "__record_id" in records.columns else []
    return [str(item) for item in options if str(item).strip()] or [NO_RENDERABLE_RECORDS_LABEL]


def build_notebook_baserender_record_choices(record_ids: Iterable[Any]) -> list[dict[str, str]]:
    """Return stable dropdown labels for renderable record ids."""

    values = normalise_record_ids(record_ids)
    if not values:
        return [{"label": NO_RENDERABLE_RECORDS_LABEL, "record_id": NO_RENDERABLE_RECORDS_LABEL}]
    return [
        {
            "label": f"{index}. {compact_record_id(record_id)}",
            "record_id": record_id,
        }
        for index, record_id in enumerate(values, start=1)
    ]


def build_notebook_baserender_record_annotation_counts(
    records_path: str | Path,
    contract: Mapping[str, Any],
    *,
    record_ids: Iterable[Any] | None = None,
) -> dict[str, int]:
    """Return per-record annotation counts for selected-sequence dropdown context."""

    if not bool(contract.get("available")):
        return {}

    import polars as pl

    identifier = id_column(contract)
    annotation = annotation_column(contract)
    if annotation is None:
        return {}
    source_columns = (
        metadata_source_columns(contract)
        if metadata_records_path(contract) is not None
        else contract_source_columns(contract)
    )
    if identifier not in source_columns:
        source_columns.append(identifier)
    if annotation not in source_columns:
        source_columns.append(annotation)
    metadata_path = metadata_records_path(contract)
    source_path = metadata_path or str(records_path)
    scan = pl.scan_parquet(source_path).select(source_columns)
    schema = scan.collect_schema()
    if metadata_path is None:
        for expr in contract_valid_filters(pl, contract, schema):
            scan = scan.filter(expr)
    else:
        scan = scan.filter(pl.col(identifier).is_not_null())
    selected_ids = normalise_record_ids(record_ids)
    if record_ids is not None:
        if not selected_ids:
            return {}
        scan = scan.filter(pl.col(identifier).cast(pl.Utf8).is_in(selected_ids))
    count_expr = annotation_count_expr(pl, annotation, schema)
    counts_df = (
        scan.select(pl.col(identifier).cast(pl.Utf8).alias("__record_id"), count_expr)
        .drop_nulls(subset=["__record_id"])
        .collect()
    )
    return {
        str(row["__record_id"]): max(0, int(row["__annotation_count"] or 0))
        for row in counts_df.to_dicts()
        if str(row["__record_id"]).strip()
    }


def build_notebook_baserender_record_choices_with_counts(
    record_ids: Iterable[Any],
    annotation_counts: Mapping[str, int],
    *,
    annotation_label: str = "annotations",
) -> list[dict[str, str]]:
    """Return dropdown labels with annotation counts while preserving selected-record identity."""

    rows = build_notebook_baserender_record_choices(record_ids)
    if not rows or rows[0]["record_id"] == NO_RENDERABLE_RECORDS_LABEL:
        return rows
    label = str(annotation_label or "annotations").strip() or "annotations"
    out: list[dict[str, str]] = []
    for row in rows:
        record_id = str(row["record_id"])
        count = max(0, int(annotation_counts.get(record_id, 0)))
        out.append(
            {
                "label": f"{row['label']} | {count} {label}",
                "record_id": record_id,
            }
        )
    return out


def select_notebook_baserender_default_record_id(
    record_ids: Iterable[Any],
    annotation_counts: Mapping[str, int] | None = None,
) -> str:
    """Choose the first annotated selected record, falling back to the first selected record."""

    values = normalise_record_ids(record_ids)
    if not values:
        return NO_RENDERABLE_RECORDS_LABEL
    counts = dict(annotation_counts or {})
    for record_id in values:
        if int(counts.get(record_id, 0)) > 0:
            return record_id
    return values[0]


def build_notebook_selected_baserender_record_ids(
    campaign_analysis: Any,
    *,
    round_value: Any | None,
    run_id: Any | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return selected record ids and compact status rows for BaseRender review."""

    selected_ids: list[str] = []
    status_rows: list[dict[str, Any]] = []
    if round_value is None:
        return selected_ids, [{"field": "selection scope", "value": "no rounds available"}]
    run_text = str(run_id or "").strip()
    if not run_text:
        return selected_ids, [{"field": "selection scope", "value": "no run available"}]

    try:
        import polars as pl

        round_int = int(round_value)
        pred_df = campaign_analysis.read_predictions(
            columns=[
                "id",
                "as_of_round",
                "run_id",
                "sel__rank_competition",
                "sel__is_selected",
            ],
            round_selector=[round_int],
            run_id=run_text,
            allow_missing=True,
        )
        selected_df = pred_df.filter(pl.col("sel__is_selected").fill_null(False)) if not pred_df.is_empty() else pred_df
        sort_columns = [column for column in ("sel__rank_competition", "id") if column in selected_df.columns]
        if sort_columns:
            selected_df = selected_df.sort(sort_columns)
        if "id" in selected_df.columns:
            selected_ids = [
                str(value)
                for value in selected_df.get_column("id").cast(pl.Utf8).drop_nulls().to_list()
                if str(value).strip()
            ]
        status_rows.extend(
            [
                {"field": "selection round", "value": round_int},
                {"field": "selection run", "value": run_text},
                {"field": "selected records", "value": len(selected_ids)},
            ]
        )
    except Exception as exc:
        status_rows.append({"field": "selection ledger", "value": f"unavailable: {exc}"})
    return selected_ids, status_rows


def load_notebook_baserender_record_row(
    records_path: str | Path,
    record_id: str,
    contract: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Load one contract-valid record row for notebook BaseRender display."""

    if not bool(contract.get("available")) or str(record_id) == NO_RENDERABLE_RECORDS_LABEL:
        return None

    import polars as pl

    identifier = id_column(contract)
    source_columns = record_source_columns(contract)
    if identifier not in source_columns:
        source_columns.append(identifier)
    scan = pl.scan_parquet(str(records_path)).select(source_columns)
    schema = scan.collect_schema()
    for expr in contract_valid_filters(pl, contract, schema):
        scan = scan.filter(expr)
    metadata_path = metadata_records_path(contract)
    if metadata_path is not None:
        scan = join_metadata_rows(pl, scan, metadata_path, id_column_name=identifier, contract=contract)
    row_df = scan.filter(pl.col(identifier).cast(pl.Utf8) == str(record_id)).limit(1).collect()
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
