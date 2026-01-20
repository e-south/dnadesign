"""Selection helpers for dashboard comparisons."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import polars as pl

from ...registries.selection import normalize_selection_result
from .util import is_altair_undefined

_OBJECTIVE_MODES = {"maximize", "minimize"}


def resolve_objective_mode(selection_params: Mapping[str, Any]) -> tuple[str, list[str]]:
    """
    Resolve objective_mode with legacy alias support and warnings.
    Returns (mode, warnings).
    """
    warnings: list[str] = []
    mode_raw = selection_params.get("objective_mode")
    legacy_raw = selection_params.get("objective")
    if mode_raw is not None and legacy_raw is not None:
        mode_str = str(mode_raw).strip().lower()
        legacy_str = str(legacy_raw).strip().lower()
        if mode_str != legacy_str:
            raise ValueError(
                "selection.params has both 'objective_mode' and legacy 'objective' with conflicting values "
                f"({mode_str!r} vs {legacy_str!r})"
            )
    if mode_raw is None and legacy_raw is not None:
        warnings.append("selection.params.objective is deprecated; prefer selection.params.objective_mode.")
        mode_raw = legacy_raw
    if mode_raw is None:
        mode_raw = "maximize"
    mode = str(mode_raw).strip().lower()
    if mode not in _OBJECTIVE_MODES:
        warnings.append(f"Unknown objective mode {mode!r}; defaulting to 'maximize'.")
        mode = "maximize"
    return mode, warnings


def coerce_selection_dataframe(selected_raw: object) -> pl.DataFrame | None:
    if selected_raw is None or is_altair_undefined(selected_raw):
        return None
    if isinstance(selected_raw, pl.DataFrame):
        return selected_raw
    try:
        return pl.from_pandas(selected_raw)
    except Exception:
        return None


def compute_selection_overlay(
    *,
    ids: np.ndarray,
    scores: np.ndarray,
    selection_params: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    mode, warnings = resolve_objective_mode(selection_params)
    try:
        top_k = int(selection_params.get("top_k", 10))
    except Exception:
        top_k = 10
    tie_handling = str(selection_params.get("tie_handling", "competition_rank"))
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
        return df, [], f"Selection reconstruction failed: missing score column '{score_col}'."

    ids = np.asarray(df.get_column(id_col).cast(pl.Utf8).to_list(), dtype=str)
    scores = df.select(pl.col(score_col).fill_null(float("nan")).cast(pl.Float64)).to_numpy().ravel()
    try:
        ranks, selected, warnings = compute_selection_overlay(
            ids=ids,
            scores=scores,
            selection_params=selection_params,
        )
    except Exception as exc:
        return df, [], f"Selection reconstruction failed: {exc}"
    df_out = df.with_columns(
        pl.Series(rank_col, ranks),
        pl.Series(top_k_col, selected),
    )
    return df_out, warnings, None
