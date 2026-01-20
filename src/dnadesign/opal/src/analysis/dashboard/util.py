"""Shared utility helpers for dashboard notebooks."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import polars as pl


def deep_as_py(x: object) -> object:
    try:
        if hasattr(x, "as_py"):
            return x.as_py()
        if hasattr(x, "to_pylist"):
            return x.to_pylist()
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        return [deep_as_py(v) for v in x.tolist()]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, pl.Series):
        return [deep_as_py(v) for v in x.to_list()]
    if isinstance(x, dict):
        return {k: deep_as_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [deep_as_py(v) for v in x]
    return x


def namespace_summary(columns: Sequence[str], max_examples: int = 3) -> pl.DataFrame:
    buckets: dict[str, list[str]] = {}
    for name in columns:
        if "__" in name:
            namespace = name.split("__", 1)[0]
        else:
            namespace = "core"
        buckets.setdefault(namespace, []).append(name)
    rows = []
    for namespace, cols in sorted(buckets.items()):
        cols_sorted = sorted(cols)
        examples = ", ".join(cols_sorted[:max_examples])
        rows.append({"namespace": namespace, "count": len(cols), "examples": examples})
    if not rows:
        return pl.DataFrame({"namespace": [], "count": [], "examples": []})
    return pl.DataFrame(rows)


def missingness_summary(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame({"column": [], "null_pct": [], "non_null_count": []})
    total = df.height
    null_counts = df.null_count()
    null_long = null_counts.transpose(
        include_header=True,
        header_name="column",
        column_names=["null_count"],
    )
    return (
        null_long.with_columns(
            (pl.col("null_count") / total * 100).alias("null_pct"),
            (pl.lit(total) - pl.col("null_count")).alias("non_null_count"),
        )
        .select(["column", "null_pct", "non_null_count"])
        .sort("null_pct", descending=True)
    )


def safe_is_numeric(dtype: pl.DataType) -> bool:
    if dtype in (pl.Null, pl.Object):
        return False
    unknown = getattr(pl, "Unknown", None)
    if unknown is not None and dtype == unknown:
        return False
    try:
        return bool(dtype.is_numeric())
    except Exception:
        return False


_COLOR_DENYLIST = {
    "__row_id",
    "id",
    "id_",
    "id__",
    "densegen__gap_fill_basis",
    "densegen__gap_fill_end",
    "densegen__gap_fill_gc_actual",
    "densegen__gap_fill_gc_max",
    "densegen__gap_fill_gc_min",
    "densegen__gap_fill_relaxed",
    "densegen__gap_fill_used",
    "densegen__library_size",
    "densegen__visual",
    "densegen__used_tfbs_detail",
}
_COLOR_DENY_PREFIXES = ("densegen__gap_fill_", "id__")


def build_color_dropdown_options(
    df: pl.DataFrame,
    *,
    extra: Sequence[str] | None = None,
    include_none: bool = False,
) -> list[str]:
    options: list[str] = []
    for name, dtype in df.schema.items():
        if name.startswith("__"):
            continue
        if name in _COLOR_DENYLIST:
            continue
        if any(name.startswith(prefix) for prefix in _COLOR_DENY_PREFIXES):
            continue
        _is_nested = False
        try:
            _is_nested = bool(getattr(dtype, "is_nested")())
        except Exception:
            _is_nested = False
        if _is_nested:
            continue
        if safe_is_numeric(dtype) or dtype in (
            pl.Boolean,
            pl.String,
            pl.Categorical,
            pl.Enum,
            pl.Date,
            pl.Datetime,
        ):
            options.append(name)
    if extra:
        for name in extra:
            if name not in options:
                options.append(name)
    if include_none:
        return ["(none)"] + options
    return options


def build_friendly_column_labels(
    *,
    score_source_label: str,
    rf_prefix: str,
    campaign_slug: str | None = None,
) -> dict[str, str]:
    labels = {
        "opal__score__scalar": f"{score_source_label} score (selected source)",
        "opal__score__rank": f"{score_source_label} rank (selected source)",
        "opal__score__top_k": f"{score_source_label} Top-K (selected source)",
        "opal__ledger__score": "Ledger score (pred__y_obj_scalar)",
        "opal__ledger__top_k": "Ledger Top-K (sel__is_selected)",
        "opal__cache__score": "Records cache score (latest_pred_scalar)",
        "opal__cache__top_k": "Records cache Top-K",
        "opal__transient__score": f"{rf_prefix} score (SFXI)",
        "opal__transient__logic_fidelity": f"{rf_prefix} logic fidelity (SFXI)",
        "opal__transient__effect_scaled": f"{rf_prefix} effect scaled (SFXI)",
        "opal__transient__rank": f"{rf_prefix} rank",
        "opal__transient__top_k": f"{rf_prefix} Top-K",
        "opal__transient__observed_event": "Observed events (ingest_y)",
        "opal__transient__sfxi_scored_label": "SFXI scored labels",
    }
    if campaign_slug:
        labels[f"opal__{campaign_slug}__latest_pred_scalar"] = "OPAL latest predicted scalar"
    return labels


def list_series_to_numpy(series: pl.Series, *, expected_len: int | None = None):
    if series.is_empty():
        return None
    series_name = series.name or "values"
    try:
        df_wide = series.to_frame(series_name).select(pl.col(series_name).list.to_struct()).unnest(series_name)
        if expected_len is not None and df_wide.width != expected_len:
            return None
        if df_wide.null_count().to_numpy().sum() > 0:
            return None
        arr = df_wide.to_numpy()
        if arr.ndim != 2:
            return None
        return arr.astype(float, copy=False)
    except Exception:
        pass

    values = series.to_list()
    if not values:
        return None
    rows = []
    for v in values:
        if v is None:
            return None
        arr = np.asarray(v, dtype=float)
        if arr.ndim != 1:
            return None
        if expected_len is not None and arr.size != expected_len:
            return None
        rows.append(arr)
    return np.vstack(rows) if rows else None


def dedupe_columns(columns: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for col in columns:
        if col in seen:
            continue
        seen.add(col)
        out.append(col)
    return out


def dedupe_exprs(exprs: Sequence[pl.Expr]) -> list[pl.Expr]:
    seen: set[str] = set()
    out: list[pl.Expr] = []
    for expr in exprs:
        name = None
        try:
            name = expr.meta.output_name()
        except Exception:
            name = None
        if name is not None:
            if name in seen:
                continue
            seen.add(name)
        out.append(expr)
    return out


def is_altair_undefined(value: object) -> bool:
    if value is None:
        return False
    try:
        import altair as alt

        if value is alt.Undefined:
            return True
    except Exception:
        pass
    cls = getattr(value, "__class__", None)
    return bool(cls is not None and cls.__name__ == "UndefinedType")
