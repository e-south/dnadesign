"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/dashboard/charts/sfxi_diagnostics_altair.py

Altair chart builders for SFXI diagnostics scatter plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence

import altair as alt
import numpy as np
import polars as pl

from ...sfxi.factorial_effects import compute_factorial_effects
from ...sfxi.state_order import STATE_ORDER, assert_state_order
from ..hues import HueOption
from ..theme import DNAD_DIAGNOSTICS_PLOT_SIZE, with_title
from ..util import list_series_to_numpy, safe_is_numeric

_POINT_SIZE_RANGE = (30, 180)


def _tooltip_fields(df: pl.DataFrame, cols: Sequence[str]) -> list[alt.Tooltip]:
    seen: set[str] = set()
    tooltips: list[alt.Tooltip] = []
    for col in cols:
        if not col or col in seen or col not in df.columns:
            continue
        seen.add(col)
        kind = "Q" if safe_is_numeric(df.schema.get(col, pl.Null)) else "N"
        tooltips.append(alt.Tooltip(f"{col}:{kind}", title=col))
    return tooltips


def _color_encoding(hue: HueOption | None) -> alt.Color | alt.UndefinedType:
    if hue is None:
        return alt.Undefined
    if hue.kind == "numeric":
        return alt.Color(
            f"{hue.key}:Q",
            title=hue.label,
            legend=alt.Legend(title=hue.label, format=".2f", tickCount=5),
        )
    return alt.Color(
        f"{hue.key}:N",
        title=hue.label,
        legend=alt.Legend(title=hue.label),
    )


def _require_numeric(df: pl.DataFrame, col: str, label: str) -> np.ndarray:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {label}")
    values = df.select(pl.col(col).cast(pl.Float64, strict=False)).to_numpy().ravel()
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must be finite.")
    return values


def make_factorial_effects_chart(
    df: pl.DataFrame,
    *,
    logic_col: str,
    size_col: str | None = None,
    label_col: str | None = None,
    title: str = "Factorial effects map",
    subtitle: str | None = None,
    plot_size: int = DNAD_DIAGNOSTICS_PLOT_SIZE,
    state_order: Sequence[str] = STATE_ORDER,
) -> alt.TopLevelMixin:
    assert_state_order(state_order)
    if df.is_empty():
        raise ValueError("Factorial effects plot requires non-empty data.")
    if logic_col not in df.columns:
        raise ValueError(f"Missing logic vector column: {logic_col}")

    vec = list_series_to_numpy(df.get_column(logic_col), expected_len=None)
    if vec is None:
        raise ValueError("Invalid logic vectors: expected list-like numeric values.")
    if vec.shape[1] < 4:
        raise ValueError("Logic vectors must have length >= 4.")

    v = vec[:, 0:4]
    a_eff, b_eff, ab_eff = compute_factorial_effects(v, state_order=state_order)
    plot_df = pl.DataFrame(
        {
            "a_effect": a_eff,
            "b_effect": b_eff,
            "ab_interaction": ab_eff,
        }
    )
    if size_col is not None and size_col in df.columns:
        plot_df = plot_df.with_columns(df.get_column(size_col).alias(size_col))
    if label_col and label_col in df.columns:
        plot_df = plot_df.with_columns(df.get_column(label_col).alias(label_col))
    if "id" in df.columns:
        plot_df = plot_df.with_columns(df.get_column("id").alias("id"))
    if "__row_id" in df.columns:
        plot_df = plot_df.with_columns(df.get_column("__row_id").alias("__row_id"))

    tooltip_cols = ["id", "__row_id", "a_effect", "b_effect", "ab_interaction"]
    if size_col:
        tooltip_cols.append(size_col)

    color = alt.Color(
        "ab_interaction:Q",
        title="AB interaction",
        legend=alt.Legend(title="AB interaction", format=".2f", tickCount=5),
    )
    enc = {
        "x": alt.X("a_effect:Q", title="A effect"),
        "y": alt.Y("b_effect:Q", title="B effect"),
        "color": color,
        "tooltip": _tooltip_fields(plot_df, tooltip_cols),
    }
    if size_col and size_col in plot_df.columns:
        enc["size"] = alt.Size(
            f"{size_col}:Q",
            legend=alt.Legend(title=size_col, format=".2f", tickCount=5),
            scale=alt.Scale(range=list(_POINT_SIZE_RANGE)),
        )

    base = (
        alt.Chart(plot_df)
        .mark_circle(opacity=0.7, stroke=None, strokeWidth=0)
        .encode(**enc)
        .properties(width=plot_size, height=plot_size)
    )
    layers: list[alt.Chart] = [base]
    if label_col and label_col in plot_df.columns:
        df_obs = plot_df.filter(pl.col(label_col).fill_null(False))
        if df_obs.height:
            layers.append(
                alt.Chart(df_obs)
                .mark_circle(
                    size=_POINT_SIZE_RANGE[1],
                    stroke="#000000",
                    strokeWidth=1.3,
                    fillOpacity=0.0,
                )
                .encode(x=alt.X("a_effect:Q"), y=alt.Y("b_effect:Q"), tooltip=_tooltip_fields(df_obs, tooltip_cols))
            )
    chart = alt.layer(*layers)
    return with_title(chart, title, subtitle)


def make_support_diagnostics_chart(
    df: pl.DataFrame,
    *,
    x_col: str,
    y_col: str,
    hue: HueOption | None = None,
    label_col: str | None = None,
    title: str = "Logic support diagnostics",
    subtitle: str | None = None,
    plot_size: int = DNAD_DIAGNOSTICS_PLOT_SIZE,
) -> alt.TopLevelMixin:
    if df.is_empty():
        raise ValueError("Support diagnostics plot requires non-empty data.")
    _require_numeric(df, x_col, x_col)
    _require_numeric(df, y_col, y_col)
    if hue is not None and hue.key not in df.columns:
        raise ValueError(f"Missing hue column: {hue.key}")

    hue_spec = hue
    tooltip_cols = ["id", "__row_id", x_col, y_col]
    if hue is not None:
        if hue.key == "opal__nearest_2_factor_logic":
            label_key = f"{hue.key}__label"
            df = df.with_columns(
                pl.col(hue.key).cast(pl.Utf8, strict=False).str.zfill(4).alias(label_key),
            )
            hue_spec = HueOption(
                key=label_key,
                label=hue.label,
                kind="categorical",
                dtype=pl.Utf8,
            )
            tooltip_cols.append(label_key)
        else:
            tooltip_cols.append(hue.key)

    base = (
        alt.Chart(df)
        .mark_circle(opacity=0.7, stroke=None, strokeWidth=0)
        .encode(
            x=alt.X(f"{x_col}:Q", title=x_col),
            y=alt.Y(f"{y_col}:Q", title=y_col),
            color=_color_encoding(hue_spec),
            tooltip=_tooltip_fields(df, tooltip_cols),
        )
        .properties(width=plot_size, height=plot_size)
    )
    layers: list[alt.Chart] = [base]
    if label_col and label_col in df.columns:
        df_obs = df.filter(pl.col(label_col).fill_null(False))
        if df_obs.height:
            layers.append(
                alt.Chart(df_obs)
                .mark_circle(size=_POINT_SIZE_RANGE[1], stroke="#000000", strokeWidth=1.3, fillOpacity=0.0)
                .encode(
                    x=alt.X(f"{x_col}:Q", title=x_col),
                    y=alt.Y(f"{y_col}:Q", title=y_col),
                    tooltip=_tooltip_fields(df_obs, tooltip_cols),
                )
            )
    chart = alt.layer(*layers)
    return with_title(chart, title, subtitle)


def make_uncertainty_chart(
    df: pl.DataFrame,
    *,
    x_col: str,
    y_col: str,
    hue: HueOption | None = None,
    label_col: str | None = None,
    title: str = "Uncertainty diagnostics",
    subtitle: str | None = None,
    plot_size: int = DNAD_DIAGNOSTICS_PLOT_SIZE,
) -> alt.TopLevelMixin:
    if df.is_empty():
        raise ValueError("Uncertainty plot requires non-empty data.")
    _require_numeric(df, x_col, x_col)
    _require_numeric(df, y_col, y_col)
    if hue is not None and hue.key not in df.columns:
        raise ValueError(f"Missing hue column: {hue.key}")

    tooltip_cols = ["id", "__row_id", x_col, y_col]
    if hue is not None:
        tooltip_cols.append(hue.key)

    base = (
        alt.Chart(df)
        .mark_circle(opacity=0.7, stroke=None, strokeWidth=0)
        .encode(
            x=alt.X(f"{x_col}:Q", title=x_col),
            y=alt.Y(f"{y_col}:Q", title=y_col),
            color=_color_encoding(hue),
            tooltip=_tooltip_fields(df, tooltip_cols),
        )
        .properties(width=plot_size, height=plot_size)
    )
    layers: list[alt.Chart] = [base]
    if label_col and label_col in df.columns:
        df_obs = df.filter(pl.col(label_col).fill_null(False))
        if df_obs.height:
            layers.append(
                alt.Chart(df_obs)
                .mark_circle(size=_POINT_SIZE_RANGE[1], stroke="#000000", strokeWidth=1.3, fillOpacity=0.0)
                .encode(
                    x=alt.X(f"{x_col}:Q", title=x_col),
                    y=alt.Y(f"{y_col}:Q", title=y_col),
                    tooltip=_tooltip_fields(df_obs, tooltip_cols),
                )
            )
    chart = alt.layer(*layers)
    return with_title(chart, title, subtitle)
