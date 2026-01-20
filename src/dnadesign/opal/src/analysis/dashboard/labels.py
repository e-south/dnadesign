"""Label history parsing and normalization for dashboard views."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl

from .diagnostics import Diagnostics
from .util import deep_as_py


def opal_labeled_mask(df: pl.DataFrame, label_hist_cols: Sequence[str]) -> pl.Series:
    if not label_hist_cols:
        return pl.Series([False] * df.height)
    exprs = [(pl.col(col).is_not_null()) & (pl.col(col).list.len().fill_null(0) > 0) for col in label_hist_cols]
    return df.select(pl.any_horizontal(exprs).alias("opal_labeled"))["opal_labeled"]


def _label_hist_sample_value(df: pl.DataFrame, label_hist_col: str) -> str | None:
    try:
        series = df.select(pl.col(label_hist_col).drop_nulls()).to_series()
    except Exception:
        return None
    if series.is_empty():
        return None
    sample = series.head(1).to_list()
    if not sample:
        return None
    try:
        return json.dumps(sample[0])
    except Exception:
        return repr(sample[0])


def _coerce_mapping(value: Any) -> dict | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    for attr in ("as_dict", "to_dict"):
        if hasattr(value, attr):
            try:
                return dict(getattr(value, attr)())
            except Exception:
                continue
    return None


def _parse_label_hist_cell(
    cell: Any,
    *,
    row_id: str,
    errors: list[dict],
    y_col_name: str,
) -> list[dict]:
    def _record_error(message: str, sample: Any | None = None) -> None:
        if len(errors) >= 5:
            return
        errors.append(
            {
                "id": row_id,
                "error": message,
                "sample": (repr(sample)[:240] if sample is not None else None),
            }
        )

    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    cell = deep_as_py(cell)
    if isinstance(cell, str):
        try:
            cell = json.loads(cell)
        except Exception as exc:
            _record_error(f"label_hist JSON parse failed: {exc}", sample=cell)
            return []
    if isinstance(cell, dict):
        entries = [cell]
    elif isinstance(cell, (list, tuple)):
        entries = list(cell)
    else:
        _record_error(f"label_hist cell must be list/dict/JSON, got {type(cell).__name__}", sample=cell)
        return []

    out: list[dict] = []
    for entry in entries:
        entry = deep_as_py(entry)
        if isinstance(entry, str):
            try:
                entry = json.loads(entry)
            except Exception as exc:
                _record_error(f"label_hist entry JSON parse failed: {exc}", sample=entry)
                continue
        entry_map = _coerce_mapping(entry)
        if entry_map is None:
            _record_error("label_hist entry must be dict-like", sample=entry)
            continue
        r = entry_map.get("r", entry_map.get("round", entry_map.get("observed_round")))
        if r is None:
            _record_error("label_hist entry missing round key ('r')", sample=entry_map)
            continue
        try:
            r_int = int(r)
        except Exception as exc:
            _record_error(f"label_hist entry round is not int: {exc}", sample=entry_map)
            continue
        y_val = entry_map.get("y", entry_map.get("y_obs", entry_map.get("value")))
        if y_val is None:
            _record_error("label_hist entry missing 'y'", sample=entry_map)
            continue
        try:
            y_list = [float(v) for v in np.asarray(y_val, dtype=float).ravel().tolist()]
        except Exception as exc:
            _record_error(f"label_hist entry 'y' not numeric: {exc}", sample=y_val)
            continue
        out.append(
            {
                "observed_round": r_int,
                "label_src": entry_map.get("src", entry_map.get("source")),
                "label_ts": entry_map.get("ts", entry_map.get("timestamp")),
                y_col_name: y_list,
            }
        )
    return out


@dataclass(frozen=True)
class LabelDiagnostics:
    status: str
    label_hist_col: str | None
    dtype: str | None
    schema: dict[str, str] | None
    sample: str | None
    rows_with_labels: int
    events_parsed: int
    errors: list[dict]
    exception: str | None
    suggested_remediation: str | None
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    message: str | None = None


@dataclass(frozen=True)
class LabelEvents:
    df: pl.DataFrame
    diag: LabelDiagnostics


def _empty_label_df(
    *,
    y_col_name: str,
    sequence_col: str | None = None,
) -> pl.DataFrame:
    empty_schema = {
        "id": pl.Utf8,
        "observed_round": pl.Int64,
        "label_src": pl.Utf8,
        "label_ts": pl.Utf8,
        y_col_name: pl.Object,
        "label_source_kind": pl.Utf8,
        "campaign_slug": pl.Utf8,
    }
    if sequence_col:
        empty_schema[sequence_col] = pl.Utf8
    return pl.DataFrame(schema=empty_schema)


def _normalize_label_events(df: pl.DataFrame, *, y_col_name: str) -> pl.DataFrame:
    if y_col_name not in df.columns:
        raise ValueError(f"Label events missing required column: {y_col_name}")
    required_cols = {"id", "observed_round", "label_src", "label_ts", y_col_name}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Label events missing required columns: {sorted(missing)}")
    df = df.with_columns(
        pl.col("id").cast(pl.Utf8),
        pl.col("observed_round").cast(pl.Int64),
        pl.col("label_src").cast(pl.Utf8),
        pl.col("label_ts").cast(pl.Utf8),
        pl.col(y_col_name).cast(pl.List(pl.Float64)),
    )
    if "label_source_kind" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("label_source_kind"))
    if "campaign_slug" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("campaign_slug"))
    return df


def build_label_events(
    *,
    df: pl.DataFrame,
    label_hist_col: str,
    y_col_name: str,
    id_col: str = "id",
    sequence_col: str = "sequence",
    campaign_slug: str | None = None,
    source_kind: str = "records",
    context: object | None = None,
) -> LabelEvents:
    suggested = "Run `opal label-hist repair` or re-ingest labels with `opal ingest-y` if label_hist is malformed."
    diag = LabelDiagnostics(
        status="ok",
        label_hist_col=label_hist_col,
        dtype=None,
        schema=None,
        sample=None,
        rows_with_labels=0,
        events_parsed=0,
        errors=[],
        exception=None,
        suggested_remediation=suggested,
        diagnostics=Diagnostics(),
        message=None,
    )
    if campaign_slug is None and context is not None:
        campaign_slug = getattr(getattr(context, "campaign_info", None), "slug", None)

    if df.is_empty():
        return LabelEvents(
            df=_empty_label_df(y_col_name=y_col_name, sequence_col=sequence_col),
            diag=LabelDiagnostics(
                **{
                    **diag.__dict__,
                    "status": "empty_df",
                    "message": "Input DataFrame is empty.",
                    "diagnostics": diag.diagnostics.add_warning("Input DataFrame is empty."),
                }
            ),
        )
    if label_hist_col not in df.columns:
        missing_msg = f"Missing label history column '{label_hist_col}'."
        return LabelEvents(
            df=_empty_label_df(y_col_name=y_col_name, sequence_col=sequence_col),
            diag=LabelDiagnostics(
                **{
                    **diag.__dict__,
                    "status": "missing_column",
                    "message": missing_msg,
                    "diagnostics": diag.diagnostics.add_error(missing_msg),
                }
            ),
        )

    diag = LabelDiagnostics(
        **{
            **diag.__dict__,
            "dtype": str(df.schema.get(label_hist_col)),
            "schema": {name: str(dtype) for name, dtype in df.schema.items()},
            "sample": _label_hist_sample_value(df, label_hist_col),
        }
    )

    select_cols = [col for col in df.columns if col != label_hist_col]
    if label_hist_col not in select_cols:
        select_cols.append(label_hist_col)

    rows: list[dict] = []
    errors: list[dict] = []
    try:
        for row in df.select(select_cols).iter_rows(named=True):
            _id = str(row.get(id_col))
            cell = row.get(label_hist_col)
            if cell is None:
                continue
            parsed = _parse_label_hist_cell(cell, row_id=_id, errors=errors, y_col_name=y_col_name)
            if parsed:
                diag = LabelDiagnostics(**{**diag.__dict__, "rows_with_labels": diag.rows_with_labels + 1})
                for entry in parsed:
                    for col in select_cols:
                        if col == label_hist_col:
                            continue
                        if col not in entry:
                            entry[col] = row.get(col)
                    entry["id"] = _id
                    entry["label_source_kind"] = source_kind
                    entry["campaign_slug"] = campaign_slug
                    rows.append(entry)
    except Exception as exc:
        error_msg = "Label history parsing failed."
        diag = LabelDiagnostics(
            **{
                **diag.__dict__,
                "status": "error",
                "message": error_msg,
                "exception": str(exc),
                "errors": errors,
                "diagnostics": diag.diagnostics.add_error(error_msg),
            }
        )
        return LabelEvents(df=_empty_label_df(y_col_name=y_col_name, sequence_col=sequence_col), diag=diag)

    diag = LabelDiagnostics(**{**diag.__dict__, "events_parsed": len(rows), "errors": errors})
    if errors:
        warn_msg = "Some label_hist cells could not be parsed."
        diag = LabelDiagnostics(
            **{
                **diag.__dict__,
                "status": "parse_warning",
                "message": warn_msg,
                "diagnostics": diag.diagnostics.add_warning(warn_msg),
            }
        )

    if not rows:
        return LabelEvents(df=_empty_label_df(y_col_name=y_col_name, sequence_col=sequence_col), diag=diag)

    df_out = pl.DataFrame(rows)
    try:
        df_out = _normalize_label_events(df_out, y_col_name=y_col_name)
    except Exception as exc:
        error_msg = "Label schema normalization failed."
        diag = LabelDiagnostics(
            **{
                **diag.__dict__,
                "status": "error",
                "message": error_msg,
                "exception": str(exc),
                "diagnostics": diag.diagnostics.add_error(error_msg),
            }
        )
        return LabelEvents(df=_empty_label_df(y_col_name=y_col_name, sequence_col=sequence_col), diag=diag)
    return LabelEvents(df=df_out, diag=diag)


def build_label_events_from_ledger(
    *,
    ledger_labels_df: pl.DataFrame,
    df_active: pl.DataFrame,
    y_col_name: str | None,
    x_col_name: str | None = None,
    campaign_slug: str | None = None,
    source_kind: str = "ledger",
    context: object | None = None,
) -> LabelEvents:
    suggested = "Ledger labels should include id/src/y_obs; regenerate ledger if missing."
    diag = LabelDiagnostics(
        status="ok",
        label_hist_col=None,
        dtype=None,
        schema=None,
        sample=None,
        rows_with_labels=0,
        events_parsed=0,
        errors=[],
        exception=None,
        suggested_remediation=suggested,
        diagnostics=Diagnostics(),
        message=None,
    )
    if campaign_slug is None and context is not None:
        campaign_slug = getattr(getattr(context, "campaign_info", None), "slug", None)
    if y_col_name is None:
        if context is not None:
            y_col_name = getattr(getattr(context, "campaign_info", None), "y_column", None)
        if y_col_name is None:
            y_col_name = "y_obs"
    if x_col_name is None and context is not None:
        x_col_name = getattr(getattr(context, "campaign_info", None), "x_column", None)

    if ledger_labels_df is None or ledger_labels_df.is_empty():
        return LabelEvents(df=_empty_label_df(y_col_name=y_col_name), diag=diag)
    if "id" not in ledger_labels_df.columns:
        missing_msg = "Ledger labels missing id column."
        diag = LabelDiagnostics(
            **{
                **diag.__dict__,
                "status": "missing_column",
                "message": missing_msg,
                "diagnostics": diag.diagnostics.add_error(missing_msg),
            }
        )
        return LabelEvents(df=_empty_label_df(y_col_name=y_col_name), diag=diag)
    if "y_obs" not in ledger_labels_df.columns:
        missing_msg = "Ledger labels missing y_obs column."
        diag = LabelDiagnostics(
            **{
                **diag.__dict__,
                "status": "missing_column",
                "message": missing_msg,
                "diagnostics": diag.diagnostics.add_error(missing_msg),
            }
        )
        return LabelEvents(df=_empty_label_df(y_col_name=y_col_name), diag=diag)

    join_cols = ["id"]
    candidate_cols = [
        "__row_id",
        "sequence",
        "cluster__ldn_v1",
        "cluster__ldn_v1__umap_x",
        "cluster__ldn_v1__umap_y",
    ]
    if x_col_name:
        candidate_cols.insert(1, x_col_name)
    for _col in candidate_cols:
        if _col and _col in df_active.columns and _col not in ledger_labels_df.columns:
            join_cols.append(_col)
    df_labels = ledger_labels_df.with_columns(pl.col("id").cast(pl.Utf8))
    df_active_join = df_active.select(join_cols).with_columns(pl.col("id").cast(pl.Utf8))
    df_join = df_labels.join(df_active_join, on="id", how="left")
    df_join = df_join.with_columns(
        [
            pl.col("y_obs").alias(y_col_name),
            pl.col("src").alias("label_src"),
            pl.lit(None).cast(pl.Utf8).alias("label_ts"),
            pl.lit(source_kind).alias("label_source_kind"),
            pl.lit(campaign_slug).alias("campaign_slug"),
        ]
    )
    diag = LabelDiagnostics(**{**diag.__dict__, "events_parsed": df_join.height, "rows_with_labels": df_join.height})
    try:
        df_join = _normalize_label_events(df_join, y_col_name=y_col_name)
    except Exception as exc:
        error_msg = "Label schema normalization failed."
        diag = LabelDiagnostics(
            **{
                **diag.__dict__,
                "status": "error",
                "message": error_msg,
                "exception": str(exc),
                "diagnostics": diag.diagnostics.add_error(error_msg),
            }
        )
        return LabelEvents(df=_empty_label_df(y_col_name=y_col_name), diag=diag)
    return LabelEvents(df=df_join, diag=diag)


def dedup_latest_labels(df: pl.DataFrame, *, id_col: str, round_col: str) -> pl.DataFrame:
    if df.is_empty():
        return df
    if round_col not in df.columns:
        return df
    return df.sort(round_col).unique(subset=[id_col], keep="last")


def observed_event_ids(df_labels: pl.DataFrame, *, label_src: str = "ingest_y") -> list[str]:
    if df_labels is None or df_labels.is_empty():
        return []
    if "id" not in df_labels.columns or "label_src" not in df_labels.columns:
        return []
    return (
        df_labels.filter(pl.col("label_src") == label_src)
        .select(pl.col("id").cast(pl.Utf8).drop_nulls().unique())
        .to_series()
        .to_list()
    )


def infer_round_from_labels(df_labels: pl.DataFrame, *, round_col: str = "observed_round") -> int | None:
    if df_labels is None or df_labels.is_empty():
        return None
    if round_col not in df_labels.columns:
        return None
    try:
        value = df_labels.select(pl.col(round_col).drop_nulls().max()).item()
    except Exception:
        return None
    if value is None:
        return None
    return int(value)
