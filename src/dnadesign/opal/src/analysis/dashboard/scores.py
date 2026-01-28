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
    if metrics_df is None or metrics_df.is_empty():
        raise ValueError("No predictions available for the selected round/run.")
    if score_col not in metrics_df.columns:
        raise ValueError(f"Missing prediction score column: {score_col}")
    if id_col not in metrics_df.columns:
        raise ValueError(f"Missing prediction id column: {id_col}")
    missing_cols: list[str] = []
    for col in [logic_col, effect_col, rank_col, top_k_col]:
        if col is not None and col not in metrics_df.columns:
            missing_cols.append(col)
    if missing_cols:
        raise ValueError(f"Missing prediction columns: {', '.join(sorted(missing_cols))}")

    select_cols = [id_col, score_col]
    rename_map = {score_col: "opal__view__score"}
    if logic_col is not None:
        select_cols.append(logic_col)
        rename_map[logic_col] = "opal__view__logic_fidelity"
    if effect_col is not None:
        select_cols.append(effect_col)
        rename_map[effect_col] = "opal__view__effect_scaled"
    if rank_col is not None:
        select_cols.append(rank_col)
        rename_map[rank_col] = "opal__view__rank"
    if top_k_col is not None:
        select_cols.append(top_k_col)
        rename_map[top_k_col] = "opal__view__top_k"

    df_metrics = metrics_df.select(select_cols).rename(rename_map)
    df_view = df_view.join(df_metrics, on=id_col, how="left")

    missing_scores = df_view.filter(pl.col("opal__view__score").is_null()).height
    if missing_scores:
        raise ValueError(
            f"Missing predictions for {missing_scores} of {df_view.height} rows "
            "in `opal__view__score`. Ensure predictions cover the full dataset "
            "for the selected run."
        )
    for view_col in ["opal__view__logic_fidelity", "opal__view__effect_scaled", "opal__view__rank"]:
        if df_view.select(pl.col(view_col).is_null().any()).item():
            raise ValueError(f"Missing values in `{view_col}`; predictions must populate all rows.")
    if df_view.select(pl.col("opal__view__top_k").is_null().any()).item():
        raise ValueError("Missing values in `opal__view__top_k`; predictions must populate all rows.")

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
        if id_col not in observed_scores_df.columns or "score" not in observed_scores_df.columns:
            raise ValueError("Observed scores missing id/score columns.")
        df_obs = observed_scores_df.select([id_col, "score"]).rename({"score": "opal__view__observed_score"})
        df_view = df_view.join(df_obs, on=id_col, how="left")
    else:
        df_view = _ensure_view_cols(df_view, ["opal__view__observed_score"])

    return ViewBundle(df=df_view, diagnostics=diag, ready=True)
