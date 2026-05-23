"""Record-contract helpers for campaign progress notebooks."""

from __future__ import annotations

from typing import Any, Mapping

import polars as pl

from ..dashboard.datasets import CampaignInfo
from .models import OPAL_RECORD_IDENTITY_COLUMNS, RecordsContractReport


def campaign_label_hist_column(info: CampaignInfo | None) -> str | None:
    if info is None:
        return None
    return f"opal__{info.slug}__label_hist"


def required_record_columns(info: CampaignInfo | None) -> tuple[str, ...]:
    columns = list(OPAL_RECORD_IDENTITY_COLUMNS)
    if info is not None and info.x_column:
        columns.append(info.x_column)
    return tuple(dict.fromkeys(columns))


def assess_records_contract(df: pl.DataFrame, info: CampaignInfo | None) -> RecordsContractReport:
    required = required_record_columns(info)
    columns = set(df.columns)
    x_column = info.x_column if info is not None else None
    return RecordsContractReport(
        row_count=int(df.height),
        column_count=len(df.columns),
        required_columns=required,
        missing_required_columns=tuple(col for col in required if col not in columns),
        x_column=x_column,
        label_hist_column=campaign_label_hist_column(info),
        x_values_loaded=bool(x_column and x_column in columns),
    )


def assess_records_contract_for_values(
    df: pl.DataFrame,
    *,
    campaign_slug: str | None,
    x_column: str | None,
) -> RecordsContractReport:
    required = list(OPAL_RECORD_IDENTITY_COLUMNS)
    if x_column:
        required.append(str(x_column))
    columns = set(df.columns)
    label_hist_column = f"opal__{campaign_slug}__label_hist" if campaign_slug else None
    return RecordsContractReport(
        row_count=int(df.height),
        column_count=len(df.columns),
        required_columns=tuple(dict.fromkeys(required)),
        missing_required_columns=tuple(col for col in dict.fromkeys(required) if col not in columns),
        x_column=str(x_column) if x_column else None,
        label_hist_column=label_hist_column,
        x_values_loaded=bool(x_column and str(x_column) in columns),
    )


def assess_records_contract_for_schema(
    *,
    row_count: int,
    schema_columns: tuple[str, ...] | list[str],
    campaign_slug: str | None,
    x_column: str | None,
    loaded_columns: tuple[str, ...] | list[str] | None = None,
) -> RecordsContractReport:
    required = list(OPAL_RECORD_IDENTITY_COLUMNS)
    if x_column:
        required.append(str(x_column))
    schema_column_set = {str(column) for column in schema_columns}
    loaded_column_set = {str(column) for column in loaded_columns or ()}
    label_hist_column = f"opal__{campaign_slug}__label_hist" if campaign_slug else None
    return RecordsContractReport(
        row_count=int(row_count),
        column_count=len(schema_column_set),
        required_columns=tuple(dict.fromkeys(required)),
        missing_required_columns=tuple(col for col in dict.fromkeys(required) if col not in schema_column_set),
        x_column=str(x_column) if x_column else None,
        label_hist_column=label_hist_column,
        x_values_loaded=bool(x_column and str(x_column) in loaded_column_set),
    )


def build_records_preview(df: pl.DataFrame, report: RecordsContractReport, *, limit: int = 25) -> pl.DataFrame:
    if df.is_empty():
        return df.head(0)

    exprs: list[pl.Expr] = []
    if "id" in df.columns:
        exprs.append(pl.col("id").cast(pl.Utf8).alias("id"))
    if "sequence" in df.columns:
        exprs.append(pl.col("sequence").cast(pl.Utf8).str.slice(0, 96).alias("sequence_preview"))
        exprs.append(pl.col("sequence").cast(pl.Utf8).str.len_chars().alias("sequence_length"))
    for col in ("bio_type", "alphabet"):
        if col in df.columns:
            exprs.append(pl.col(col))
    if report.x_column and report.x_column in df.columns:
        exprs.append(pl.col(report.x_column).is_not_null().alias("x_present"))
    if report.label_hist_column and report.label_hist_column in df.columns:
        exprs.append(pl.col(report.label_hist_column).is_not_null().alias("label_hist_present"))
    if not exprs:
        return df.head(limit)
    return df.select(exprs).head(limit)


def records_status_rows(report: RecordsContractReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {"field": "rows", "value": report.row_count},
        {"field": "columns", "value": report.column_count},
        {"field": "required OPAL columns", "value": ", ".join(report.required_columns)},
        {
            "field": "records contract",
            "value": "ready" if report.ready else "missing required columns",
        },
    ]
    if not report.ready:
        rows.append({"field": "missing columns", "value": ", ".join(report.missing_required_columns)})
    if report.label_hist_column:
        rows.append({"field": "campaign label history column", "value": report.label_hist_column})
    if report.x_column:
        rows.append({"field": "X values loaded in this view", "value": "yes" if report.x_values_loaded else "no"})
    return rows


def records_status_lines(report: RecordsContractReport) -> list[str]:
    return [f"- {row['field']}: `{row['value']}`" for row in records_status_rows(report)]


def active_record_rows(row: Mapping[str, Any], report: RecordsContractReport) -> list[dict[str, Any]]:
    sequence = str(row.get("sequence") or "")
    rows: list[dict[str, Any]] = [
        {"field": "id", "value": row.get("id")},
        {"field": "sequence length", "value": len(sequence)},
    ]
    if report.x_column:
        rows.append({"field": "X present", "value": row.get(report.x_column) is not None})
    if report.label_hist_column:
        rows.append({"field": "label history present", "value": row.get(report.label_hist_column) is not None})
    if sequence:
        rows.append({"field": "sequence preview", "value": sequence[:120]})
    return rows
