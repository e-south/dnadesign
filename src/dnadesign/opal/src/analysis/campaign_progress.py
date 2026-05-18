"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/campaign_progress.py

Small helpers for read-only OPAL campaign progress notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import polars as pl

from .dashboard.datasets import CampaignInfo

OPAL_RECORD_IDENTITY_COLUMNS = ("id", "bio_type", "sequence", "alphabet")
DEFAULT_PROJECTION_COLUMNS = ("cluster__ldn_v1__umap_x", "cluster__ldn_v1__umap_y")


@dataclass(frozen=True)
class RecordsContractReport:
    row_count: int
    column_count: int
    required_columns: tuple[str, ...]
    missing_required_columns: tuple[str, ...]
    x_column: str | None
    label_hist_column: str | None
    projection_columns_present: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.missing_required_columns

    @property
    def projection_ready(self) -> bool:
        return all(col in self.projection_columns_present for col in DEFAULT_PROJECTION_COLUMNS)


@dataclass(frozen=True)
class OptionalTableRead:
    name: str
    path: Path | None
    df: pl.DataFrame
    status: str
    message: str

    @property
    def available(self) -> bool:
        return self.status == "available"


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
    projection_present = tuple(col for col in DEFAULT_PROJECTION_COLUMNS if col in columns)
    return RecordsContractReport(
        row_count=int(df.height),
        column_count=len(df.columns),
        required_columns=required,
        missing_required_columns=tuple(col for col in required if col not in columns),
        x_column=info.x_column if info is not None else None,
        label_hist_column=campaign_label_hist_column(info),
        projection_columns_present=projection_present,
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
    projection_present = tuple(col for col in DEFAULT_PROJECTION_COLUMNS if col in columns)
    label_hist_column = f"opal__{campaign_slug}__label_hist" if campaign_slug else None
    return RecordsContractReport(
        row_count=int(df.height),
        column_count=len(df.columns),
        required_columns=tuple(dict.fromkeys(required)),
        missing_required_columns=tuple(col for col in dict.fromkeys(required) if col not in columns),
        x_column=str(x_column) if x_column else None,
        label_hist_column=label_hist_column,
        projection_columns_present=projection_present,
    )


def records_status_lines(report: RecordsContractReport) -> list[str]:
    lines = [
        f"- Rows: `{report.row_count}`",
        f"- Columns: `{report.column_count}`",
        f"- Required OPAL columns: {', '.join(f'`{col}`' for col in report.required_columns)}",
    ]
    if report.ready:
        lines.append("- Records contract: **ready**")
    else:
        missing = ", ".join(f"`{col}`" for col in report.missing_required_columns)
        lines.append(f"- Records contract: **missing required columns**: {missing}")
    if report.label_hist_column:
        lines.append(f"- Campaign label history column: `{report.label_hist_column}`")
    return lines


def projection_status_lines(report: RecordsContractReport) -> list[str]:
    if report.projection_ready:
        return [
            "- Projection context: **available**",
            "- Projection columns are optional OPAL context, not campaign readiness requirements.",
        ]
    missing = [col for col in DEFAULT_PROJECTION_COLUMNS if col not in report.projection_columns_present]
    return [
        "- Projection context: **not available**",
        "- OPAL records remain inspectable without LatentDNA/UMAP columns.",
        "- Missing optional projection columns: " + ", ".join(f"`{col}`" for col in missing),
    ]


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
    for col in DEFAULT_PROJECTION_COLUMNS:
        if col in df.columns:
            exprs.append(pl.col(col))

    if not exprs:
        return df.head(limit)
    return df.select(exprs).head(limit)


def _parquet_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(pl.scan_parquet(str(path)).select(pl.len()).collect().item())
    except Exception:
        return None


def build_ledger_status_table(workdir: Path | None) -> pl.DataFrame:
    if workdir is None:
        return pl.DataFrame(
            {
                "artifact": ["state", "labels", "runs", "predictions"],
                "status": ["missing workdir"] * 4,
                "rows": [None] * 4,
                "path": [""] * 4,
            }
        )

    state_path = workdir / "state.json"
    labels_path = workdir / "outputs" / "ledger" / "labels.parquet"
    runs_path = workdir / "outputs" / "ledger" / "runs.parquet"
    predictions_dir = workdir / "outputs" / "ledger" / "predictions"
    prediction_parts = sorted(predictions_dir.glob("*.parquet")) if predictions_dir.exists() else []
    rows = [
        {
            "artifact": "state",
            "status": "present" if state_path.exists() else "missing",
            "rows": None,
            "path": str(state_path),
        },
        {
            "artifact": "labels",
            "status": "present" if labels_path.exists() else "missing",
            "rows": _parquet_row_count(labels_path),
            "path": str(labels_path),
        },
        {
            "artifact": "runs",
            "status": "present" if runs_path.exists() else "missing",
            "rows": _parquet_row_count(runs_path),
            "path": str(runs_path),
        },
        {
            "artifact": "predictions",
            "status": "present" if prediction_parts else "missing",
            "rows": len(prediction_parts) if prediction_parts else None,
            "path": str(predictions_dir),
        },
    ]
    return pl.DataFrame(rows)


def read_optional_table(
    name: str,
    path: Path | str | None,
    loader: Callable[[], pl.DataFrame],
) -> OptionalTableRead:
    table_path = Path(path) if path is not None else None
    try:
        df = loader()
    except Exception as exc:
        return OptionalTableRead(
            name=str(name),
            path=table_path,
            df=pl.DataFrame(),
            status="unavailable",
            message=str(exc),
        )
    status = "available"
    message = "available"
    if df.is_empty():
        status = "empty"
        message = "table exists but has zero rows"
    return OptionalTableRead(
        name=str(name),
        path=table_path,
        df=df,
        status=status,
        message=message,
    )


def unavailable_table(
    name: str,
    path: Path | str | None,
    message: str,
) -> OptionalTableRead:
    table_path = Path(path) if path is not None else None
    return OptionalTableRead(
        name=str(name),
        path=table_path,
        df=pl.DataFrame(),
        status="unavailable",
        message=str(message),
    )


def table_status_lines(table: OptionalTableRead) -> list[str]:
    path_text = str(table.path) if table.path is not None else ""
    return [
        f"- {table.name}: **{table.status}**",
        f"- Rows: `{table.df.height}`",
        f"- Path: `{path_text}`",
        f"- Detail: {table.message}",
    ]


def cli_handoff_lines(config_path: Path | str) -> list[str]:
    config_text = str(config_path)
    return [
        "### Canonical OPAL inspection commands",
        "",
        "Pre-run campaign viewer generation (writes notebook):",
        "",
        "```bash",
        f"uv run opal validate -c {config_text}",
        f"uv run opal notebook generate -c {config_text} --round latest --force",
        f"uv run opal notebook run -c {config_text}",
        "```",
        "",
        "Post-run ledger inspection:",
        "",
        "```bash",
        f"uv run opal status -c {config_text} --with-ledger",
        f"uv run opal runs list -c {config_text}",
        f"uv run opal record-show -c {config_text} --selected-rank 1 --round latest --run-id latest",
        f"uv run opal verify-outputs -c {config_text} --round latest",
        f"uv run opal plot -c {config_text}",
        "```",
    ]
