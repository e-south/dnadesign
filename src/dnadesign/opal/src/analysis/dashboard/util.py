"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/dashboard/util.py

Shared utility helpers for dashboard notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

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
