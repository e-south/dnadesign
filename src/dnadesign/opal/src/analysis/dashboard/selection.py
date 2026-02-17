"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/dashboard/selection.py

Selection helpers for dashboard comparisons.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import polars as pl

from ...core.selection_contracts import (
    resolve_selection_objective_mode,
    resolve_selection_tie_handling,
)
from ...registries.selection import normalize_selection_result
from .util import is_altair_undefined


def resolve_objective_mode(
    selection_params: Mapping[str, Any],
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    mode = resolve_selection_objective_mode(selection_params, error_cls=ValueError)
    return mode, warnings


def coerce_selection_dataframe(selected_raw: object) -> pl.DataFrame | None:
    if selected_raw is None or is_altair_undefined(selected_raw):
        return None
    if isinstance(selected_raw, pl.DataFrame):
        return selected_raw
    try:
        return pl.from_pandas(selected_raw)
    except (TypeError, ValueError, pl.exceptions.PolarsError):
        return None


def resolve_brush_selection(
    *,
    df_plot: pl.DataFrame,
    selected_raw: object,
    selection_enabled: bool,
    id_col: str = "__row_id",
) -> tuple[pl.DataFrame, str]:
    if not selection_enabled:
        return df_plot.head(0), "UMAP missing; selection disabled."
    if selected_raw is None or is_altair_undefined(selected_raw):
        return df_plot.head(0), "No points selected."
    selected_df = coerce_selection_dataframe(selected_raw)
    if selected_df is None or id_col not in selected_df.columns:
        return df_plot.head(0), "Selection missing row ids."
    selected_ids = selected_df.select(pl.col(id_col).drop_nulls().unique()).to_series().to_list()
    if not selected_ids:
        return df_plot.head(0), "No points selected."
    df_selected = df_plot.filter(pl.col(id_col).is_in(selected_ids))
    return df_selected, f"Selected rows: `{df_selected.height}`"


def compute_selection_overlay(
    *,
    ids: np.ndarray,
    scores: np.ndarray,
    selection_params: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    mode, warnings = resolve_objective_mode(selection_params)
    if "top_k" not in selection_params:
        raise ValueError("selection.params.top_k is required.")
    top_k = int(selection_params.get("top_k"))
    if top_k <= 0:
        raise ValueError("selection.params.top_k must be > 0.")
    tie_handling = resolve_selection_tie_handling(selection_params, error_cls=ValueError)
    result = normalize_selection_result(
        {},
        ids=ids,
        scores=scores,
        top_k=top_k,
        tie_handling=tie_handling,
        objective=mode,
    )
    return result["ranks"], result["selected_bool"], warnings


def ensure_selection_columns(
    df: pl.DataFrame,
    *,
    id_col: str,
    score_col: str,
    selection_params: Mapping[str, Any],
    rank_col: str,
    top_k_col: str,
) -> tuple[pl.DataFrame, list[str], str | None]:
    if df.is_empty():
        return df, [], "Selection reconstruction skipped (no rows)."
    if rank_col in df.columns and top_k_col in df.columns:
        return df, [], None
    if id_col not in df.columns:
        return df, [], f"Selection reconstruction failed: missing id column '{id_col}'."
    if score_col not in df.columns:
        return (
            df,
            [],
            f"Selection reconstruction failed: missing score column '{score_col}'.",
        )

    ids = np.asarray(df.get_column(id_col).cast(pl.Utf8).to_list(), dtype=str)
    scores = df.select(pl.col(score_col).fill_null(float("nan")).cast(pl.Float64)).to_numpy().ravel()
    try:
        ranks, selected, warnings = compute_selection_overlay(
            ids=ids,
            scores=scores,
            selection_params=selection_params,
        )
    except (TypeError, ValueError) as exc:
        return df, [], f"Selection reconstruction failed: {exc}"
    df_out = df.with_columns(
        pl.Series(rank_col, ranks),
        pl.Series(top_k_col, selected),
    )
    return df_out, warnings, None
