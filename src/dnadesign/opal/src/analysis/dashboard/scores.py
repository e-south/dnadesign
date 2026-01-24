"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/dashboard/scores.py

Canonical/overlay view helpers for dashboard plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import polars as pl

from .diagnostics import Diagnostics


@dataclass(frozen=True)
class ViewBundle:
    df: pl.DataFrame
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    ready: bool = False


def _ensure_view_cols(df: pl.DataFrame, cols: Iterable[str]) -> pl.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if not missing:
        return df
    return df.with_columns([pl.lit(None).alias(col) for col in missing])


def build_mode_view(
    *,
    df_base: pl.DataFrame,
    metrics_df: pl.DataFrame | None,
    id_col: str,
    observed_ids: set[str],
    observed_scores_df: pl.DataFrame | None,
    score_col: str,
    logic_col: str | None,
    effect_col: str | None,
    rank_col: str | None,
    top_k_col: str | None,
) -> ViewBundle:
    diag = Diagnostics()
    df_view = df_base
    ready = True

    if metrics_df is None or metrics_df.is_empty():
        diag = diag.add_error("No predictions available for the selected round/run.")
        ready = False
    elif score_col not in metrics_df.columns:
        diag = diag.add_error(f"Missing prediction score column: {score_col}")
        ready = False
    elif id_col not in metrics_df.columns:
        diag = diag.add_error(f"Missing prediction id column: {id_col}")
        ready = False

    if ready:
        select_cols = [id_col, score_col]
        rename_map = {score_col: "opal__view__score"}
        if logic_col and logic_col in metrics_df.columns:
            select_cols.append(logic_col)
            rename_map[logic_col] = "opal__view__logic_fidelity"
        if effect_col and effect_col in metrics_df.columns:
            select_cols.append(effect_col)
            rename_map[effect_col] = "opal__view__effect_scaled"
        if rank_col and rank_col in metrics_df.columns:
            select_cols.append(rank_col)
            rename_map[rank_col] = "opal__view__rank"
        if top_k_col and top_k_col in metrics_df.columns:
            select_cols.append(top_k_col)
            rename_map[top_k_col] = "opal__view__top_k"

        df_metrics = metrics_df.select(select_cols).rename(rename_map)
        df_view = df_view.join(df_metrics, on=id_col, how="left")

    df_view = _ensure_view_cols(
        df_view,
        [
            "opal__view__score",
            "opal__view__logic_fidelity",
            "opal__view__effect_scaled",
            "opal__view__rank",
            "opal__view__top_k",
        ],
    )

    if id_col in df_view.columns:
        df_view = df_view.with_columns(
            pl.col(id_col).cast(pl.Utf8).is_in(sorted(observed_ids)).alias("opal__view__observed")
        )
    else:
        df_view = df_view.with_columns(pl.lit(False).alias("opal__view__observed"))

    df_view = df_view.with_columns(
        pl.when(~pl.col("opal__view__observed"))
        .then(pl.col("opal__view__score"))
        .otherwise(None)
        .alias("opal__view__pred_score_unlabeled"),
    )

    if observed_scores_df is not None and not observed_scores_df.is_empty():
        if id_col in observed_scores_df.columns and "score" in observed_scores_df.columns:
            df_obs = observed_scores_df.select([id_col, "score"]).rename({"score": "opal__view__observed_score"})
            df_view = df_view.join(df_obs, on=id_col, how="left")
        else:
            diag = diag.add_warning("Observed scores missing id/score columns; observed_score disabled.")
            df_view = _ensure_view_cols(df_view, ["opal__view__observed_score"])
    else:
        df_view = _ensure_view_cols(df_view, ["opal__view__observed_score"])

    return ViewBundle(df=df_view, diagnostics=diag, ready=ready)
