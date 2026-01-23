"""Label history parsing and normalization for dashboard views."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl

from .datasets import RoundOptions
from .diagnostics import Diagnostics
from .util import deep_as_py


def _cell_has_label(cell: Any) -> bool:
    parsed = _parse_label_hist_cell(cell, row_id="__row__", errors=[], y_col_name="y_obs")
    return bool(parsed)


def opal_labeled_mask(df: pl.DataFrame, label_hist_cols: Sequence[str]) -> pl.Series:
    if not label_hist_cols or df.is_empty():
        return pl.Series([False] * df.height)
    masks: list[pl.Series] = []
    for col in label_hist_cols:
        if col not in df.columns:
            continue
        series = df.get_column(col)
        mask = series.map_elements(_cell_has_label, return_dtype=pl.Boolean).fill_null(False)
        masks.append(mask)
    if not masks:
        return pl.Series([False] * df.height)
    out = masks[0]
    for mask in masks[1:]:
        out = out | mask
    return out


def build_round_options_from_label_hist(
    *,
    label_events_df: pl.DataFrame | None,
    pred_events_df: pl.DataFrame | None,
) -> RoundOptions:
    diagnostics = Diagnostics()
    rounds: set[int] = set()
    run_ids_by_round: dict[int, list[str]] = {}
    source = "label_hist"

    if label_events_df is not None and not label_events_df.is_empty():
        if "observed_round" in label_events_df.columns:
            try:
                values = label_events_df.select(pl.col("observed_round").drop_nulls().unique()).to_series().to_list()
                for val in values:
                    try:
                        rounds.add(int(val))
                    except Exception:
                        continue
            except Exception:
                diagnostics = diagnostics.add_warning("Failed to parse observed_round values from label history.")
        else:
            diagnostics = diagnostics.add_warning("Label history missing observed_round; label rounds unavailable.")

    if pred_events_df is not None and not pred_events_df.is_empty():
        if "as_of_round" in pred_events_df.columns:
            try:
                values = pred_events_df.select(pl.col("as_of_round").drop_nulls().unique()).to_series().to_list()
                for val in values:
                    try:
                        rounds.add(int(val))
                    except Exception:
                        continue
            except Exception:
                diagnostics = diagnostics.add_warning("Failed to parse as_of_round values from prediction history.")
        else:
            diagnostics = diagnostics.add_warning(
                "Prediction history missing as_of_round; prediction rounds unavailable."
            )

        if "run_id" in pred_events_df.columns:
            for round_val in sorted(rounds):
                try:
                    run_vals = (
                        pred_events_df.filter(pl.col("as_of_round") == int(round_val))
                        .select(pl.col("run_id").drop_nulls().unique())
                        .to_series()
                        .to_list()
                    )
                except Exception:
                    run_vals = []
                run_ids_by_round[int(round_val)] = sorted({str(v) for v in run_vals if v is not None})
        else:
            diagnostics = diagnostics.add_warning("Prediction history missing run_id; run selector disabled.")

    round_list = sorted(rounds)
    if not round_list:
        diagnostics = diagnostics.add_warning("No rounds found in label history.")

    return RoundOptions(
        rounds=round_list,
        run_ids_by_round=run_ids_by_round,
        source=source,
        diagnostics=diagnostics,
    )


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
        kind = entry_map.get("kind")
        if kind is not None and str(kind).strip().lower() != "label":
            continue
        r = entry_map.get("observed_round", entry_map.get("r", entry_map.get("round")))
        if r is None:
            _record_error("label_hist entry missing observed_round", sample=entry_map)
            continue
        try:
            r_int = int(r)
        except Exception as exc:
            _record_error(f"label_hist entry round is not int: {exc}", sample=entry_map)
            continue
        y_val = entry_map.get("y_obs", entry_map.get("y", entry_map.get("value")))
        if y_val is None:
            _record_error("label_hist entry missing y_obs", sample=entry_map)
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


@dataclass(frozen=True)
class PredDiagnostics:
    status: str
    label_hist_col: str | None
    rows_with_preds: int
    events_parsed: int
    errors: list[dict]
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    message: str | None = None


@dataclass(frozen=True)
class PredEvents:
    df: pl.DataFrame
    diag: PredDiagnostics


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


def _parse_pred_hist_cell(
    cell: Any,
    *,
    row_id: str,
    errors: list[dict],
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
        kind = entry_map.get("kind")
        if kind is None or str(kind).strip().lower() != "pred":
            continue
        as_of_round = entry_map.get("as_of_round", entry_map.get("r"))
        run_id = entry_map.get("run_id")
        if as_of_round is None or run_id is None:
            _record_error("pred entry missing as_of_round or run_id", sample=entry_map)
            continue
        try:
            round_int = int(as_of_round)
        except Exception as exc:
            _record_error(f"pred entry as_of_round not int: {exc}", sample=entry_map)
            continue
        y_hat = entry_map.get("y_hat")
        if y_hat is None:
            _record_error("pred entry missing y_hat", sample=entry_map)
            continue
        try:
            y_hat_list = [float(v) for v in np.asarray(y_hat, dtype=float).ravel().tolist()]
        except Exception as exc:
            _record_error(f"pred entry y_hat not numeric: {exc}", sample=y_hat)
            continue

        metrics = _coerce_mapping(entry_map.get("metrics") or {})
        selection = _coerce_mapping(entry_map.get("selection") or {})
        score_val = metrics.get("score") if metrics is not None else None
        if score_val is None:
            _record_error("pred entry missing metrics.score", sample=entry_map)
            continue
        try:
            score_val = float(score_val)
        except Exception as exc:
            _record_error(f"pred entry metrics.score not float: {exc}", sample=entry_map)
            continue

        rank_val = selection.get("rank") if selection is not None else None
        top_k_val = selection.get("top_k") if selection is not None else None
        if rank_val is None or top_k_val is None:
            _record_error("pred entry missing selection.rank/top_k", sample=entry_map)
            continue
        try:
            rank_val = int(rank_val)
        except Exception as exc:
            _record_error(f"pred entry selection.rank not int: {exc}", sample=entry_map)
            continue
        top_k_val = bool(top_k_val)

        objective = _coerce_mapping(entry_map.get("objective") or {})
        out.append(
            {
                "as_of_round": round_int,
                "run_id": str(run_id),
                "pred_ts": entry_map.get("ts", entry_map.get("timestamp")),
                "pred_y_hat": y_hat_list,
                "pred_score": score_val,
                "pred_logic_fidelity": metrics.get("logic_fidelity") if metrics else None,
                "pred_effect_scaled": metrics.get("effect_scaled") if metrics else None,
                "pred_effect_raw": metrics.get("effect_raw") if metrics else None,
                "pred_rank": rank_val,
                "pred_top_k": top_k_val,
                "pred_objective_name": objective.get("name") if objective else None,
                "pred_objective_params": objective.get("params") if objective else None,
            }
        )
    return out


def _empty_pred_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "id": pl.Utf8,
            "__row_id": pl.Int64,
            "as_of_round": pl.Int64,
            "run_id": pl.Utf8,
            "pred_ts": pl.Utf8,
            "pred_y_hat": pl.Object,
            "pred_score": pl.Float64,
            "pred_logic_fidelity": pl.Float64,
            "pred_effect_scaled": pl.Float64,
            "pred_effect_raw": pl.Float64,
            "pred_rank": pl.Int64,
            "pred_top_k": pl.Boolean,
            "pred_objective_name": pl.Utf8,
            "pred_objective_params": pl.Object,
        }
    )


def build_pred_events(
    *,
    df: pl.DataFrame,
    label_hist_col: str,
    id_col: str = "id",
) -> PredEvents:
    diag = PredDiagnostics(
        status="ok",
        label_hist_col=label_hist_col,
        rows_with_preds=0,
        events_parsed=0,
        errors=[],
        diagnostics=Diagnostics(),
        message=None,
    )
    if df.is_empty():
        return PredEvents(
            df=_empty_pred_df(),
            diag=PredDiagnostics(
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
        return PredEvents(
            df=_empty_pred_df(),
            diag=PredDiagnostics(
                **{
                    **diag.__dict__,
                    "status": "missing_column",
                    "message": missing_msg,
                    "diagnostics": diag.diagnostics.add_error(missing_msg),
                }
            ),
        )

    select_cols = [id_col, label_hist_col]
    if "__row_id" in df.columns and "__row_id" not in select_cols:
        select_cols.append("__row_id")

    rows: list[dict] = []
    errors: list[dict] = []
    try:
        for row in df.select(select_cols).iter_rows(named=True):
            _id = str(row.get(id_col))
            cell = row.get(label_hist_col)
            if cell is None:
                continue
            parsed = _parse_pred_hist_cell(cell, row_id=_id, errors=errors)
            if parsed:
                diag = PredDiagnostics(**{**diag.__dict__, "rows_with_preds": diag.rows_with_preds + 1})
                for entry in parsed:
                    entry["id"] = _id
                    if "__row_id" in row:
                        entry["__row_id"] = row.get("__row_id")
                    rows.append(entry)
    except Exception:
        error_msg = "Prediction history parsing failed."
        diag = PredDiagnostics(
            **{
                **diag.__dict__,
                "status": "error",
                "message": error_msg,
                "diagnostics": diag.diagnostics.add_error(error_msg),
                "errors": errors,
            }
        )
        return PredEvents(df=_empty_pred_df(), diag=diag)

    diag = PredDiagnostics(**{**diag.__dict__, "events_parsed": len(rows), "errors": errors})
    if errors:
        warn_msg = "Some pred entries could not be parsed."
        diag = PredDiagnostics(
            **{
                **diag.__dict__,
                "status": "parse_warning",
                "message": warn_msg,
                "diagnostics": diag.diagnostics.add_warning(warn_msg),
            }
        )
    if not rows:
        return PredEvents(df=_empty_pred_df(), diag=diag)

    df_out = pl.DataFrame(rows).with_columns(
        pl.col("pred_score").cast(pl.Float64),
        pl.col("pred_logic_fidelity").cast(pl.Float64),
        pl.col("pred_effect_scaled").cast(pl.Float64),
        pl.col("pred_effect_raw").cast(pl.Float64),
        pl.col("pred_rank").cast(pl.Int64),
        pl.col("pred_top_k").cast(pl.Boolean),
    )
    return PredEvents(df=df_out, diag=diag)


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
