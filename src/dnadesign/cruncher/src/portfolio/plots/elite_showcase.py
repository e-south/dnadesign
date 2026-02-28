"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/portfolio/plots/elite_showcase.py

Render cross-workspace elite showcase panels from portfolio handoff tables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd

from dnadesign.baserender import (
    cruncher_showcase_style_overrides,
    render_record_grid_figure,
)
from dnadesign.cruncher.analysis.plots._savefig import savefig
from dnadesign.cruncher.analysis.plots.elites_showcase import (
    _overlay_line_char_budget,
    _wrap_overlay_score_tokens,
    _wrap_overlay_text_line,
    build_elites_showcase_records,
)

_NORM_SCORE_EPS = 1.0e-9


def _ensure_required_columns(df: pd.DataFrame, required: list[str], *, context: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def _portfolio_overlay_text(
    elite_row: pd.Series,
    *,
    tf_names: list[str],
    ncols: int,
    max_chars: int = 80,
) -> str:
    source_label = str(elite_row.get("source_label") or elite_row.get("source_id") or "").strip()
    if not source_label:
        raise ValueError("Portfolio elite showcase requires source_label or source_id for overlay titles.")
    elite_id = str(elite_row.get("id") or elite_row.get("elite_id") or "").strip()
    if not elite_id:
        raise ValueError("Portfolio elite showcase requires elite id for overlay titles.")
    sequence_length: int | None = None
    sequence_length_value = elite_row.get("sequence_length")
    if sequence_length_value is not None and not pd.isna(sequence_length_value):
        parsed_length = pd.to_numeric(sequence_length_value, errors="coerce")
        if pd.notna(parsed_length):
            seq_len_int = int(parsed_length)
            if seq_len_int > 0:
                sequence_length = seq_len_int
    if sequence_length is None:
        sequence_text = str(elite_row.get("sequence") or "").strip()
        if sequence_text:
            sequence_length = len(sequence_text)

    elite_title = elite_id if sequence_length is None else f"{elite_id} (L={sequence_length})"
    line_chars = _overlay_line_char_budget(ncols=int(ncols), max_chars=int(max_chars))
    score_tokens: list[str] = []
    for tf_name in tf_names:
        score_column = f"norm_{tf_name}"
        if score_column not in elite_row.index:
            raise ValueError(f"Portfolio elite showcase missing required score column: {score_column}")
        score_value = pd.to_numeric(elite_row.get(score_column), errors="coerce")
        if pd.isna(score_value):
            raise ValueError(
                f"Portfolio elite showcase score column must be numeric: source={source_label!r} "
                f"elite_id={elite_id!r} column={score_column!r}"
            )
        score_tokens.append(f"{tf_name}={float(score_value):.2f}")
    source_lines = _wrap_overlay_text_line(source_label, line_chars=line_chars)
    elite_lines = _wrap_overlay_text_line(elite_title, line_chars=line_chars)
    score_lines = _wrap_overlay_score_tokens(score_tokens, line_chars=line_chars)
    return "\n".join([*source_lines, *elite_lines, *score_lines])


def _validate_normalized_series(series: pd.Series, *, context: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{context} must be numeric.")
    below = numeric < -_NORM_SCORE_EPS
    above = numeric > 1.0 + _NORM_SCORE_EPS
    if bool(below.any()) or bool(above.any()):
        raise ValueError(f"{context} must be normalized in [0,1].")
    return numeric.astype(float).clip(lower=0.0, upper=1.0)


def _source_showcase_frames(
    *,
    selected_elites_df: pd.DataFrame,
    handoff_df: pd.DataFrame,
    source_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    source_elites = selected_elites_df[selected_elites_df["source_id"].astype(str) == source_id].copy()
    if source_elites.empty:
        raise ValueError(f"Portfolio elite showcase has no selected elites for source {source_id!r}.")

    elite_ids = source_elites["elite_id"].astype(str).tolist()
    source_windows = handoff_df[
        (handoff_df["source_id"].astype(str) == source_id) & (handoff_df["elite_id"].astype(str).isin(elite_ids))
    ].copy()
    if source_windows.empty:
        raise ValueError(f"Portfolio elite showcase has no handoff windows for source {source_id!r}.")
    _ensure_required_columns(
        source_windows,
        ["elite_id", "tf", "best_start", "best_end", "best_strand", "best_score_norm"],
        context=f"Portfolio elite showcase windows ({source_id})",
    )

    key_counts = source_windows.groupby(["elite_id", "tf"]).size()
    duplicate = key_counts[key_counts > 1]
    if not duplicate.empty:
        labels = [f"({elite_id},{tf})x{int(count)}" for (elite_id, tf), count in duplicate.items()]
        raise ValueError(
            f"Portfolio elite showcase has duplicate elite/tf windows for source {source_id!r}: " + ", ".join(labels)
        )

    tf_names = sorted(source_windows["tf"].astype(str).unique().tolist())
    if not tf_names:
        raise ValueError(f"Portfolio elite showcase found no TF rows for source {source_id!r}.")

    source_windows["best_start"] = pd.to_numeric(source_windows["best_start"], errors="coerce")
    source_windows["best_end"] = pd.to_numeric(source_windows["best_end"], errors="coerce")
    if source_windows["best_start"].isna().any() or source_windows["best_end"].isna().any():
        raise ValueError(f"Portfolio elite showcase start/end columns must be numeric for source {source_id!r}.")
    source_windows["best_start"] = source_windows["best_start"].astype(int)
    source_windows["best_end"] = source_windows["best_end"].astype(int)
    source_windows["pwm_width"] = source_windows["best_end"] - source_windows["best_start"]
    if (source_windows["pwm_width"] < 1).any():
        raise ValueError(f"Portfolio elite showcase has non-positive window widths for source {source_id!r}.")
    source_windows["best_score_norm"] = _validate_normalized_series(
        source_windows["best_score_norm"],
        context=f"Portfolio elite showcase best_score_norm for source {source_id!r}",
    )

    score_wide = (
        source_windows.pivot(index="elite_id", columns="tf", values="best_score_norm")
        .rename(columns=lambda tf_name: f"norm_{tf_name}")
        .reset_index()
    )
    source_elites["elite_id"] = source_elites["elite_id"].astype(str)
    source_elites = source_elites.merge(score_wide, how="left", on="elite_id", validate="one_to_one")
    source_elites["id"] = source_elites["elite_id"]
    for tf_name in tf_names:
        score_column = f"norm_{tf_name}"
        if score_column not in source_elites.columns:
            raise ValueError(
                f"Portfolio elite showcase failed to construct score columns for source {source_id!r}: "
                f"missing {score_column}"
            )
        source_elites[score_column] = _validate_normalized_series(
            source_elites[score_column],
            context=f"Portfolio elite showcase score column {score_column!r} for source {source_id!r}",
        )

    hits_df = source_windows[["elite_id", "tf", "best_start", "pwm_width", "best_strand"]].copy()
    return source_elites, hits_df, tf_names


def plot_portfolio_elite_showcase(
    *,
    selected_elites_df: pd.DataFrame,
    handoff_df: pd.DataFrame,
    pwms_by_source: Mapping[str, Mapping[str, object]],
    out_path: Path,
    ncols: int,
    dpi: int,
) -> None:
    if selected_elites_df.empty:
        raise ValueError("Portfolio elite showcase requires at least one selected elite.")
    _ensure_required_columns(
        selected_elites_df,
        ["source_id", "source_label", "elite_id", "sequence"],
        context="selected_elites_df",
    )
    _ensure_required_columns(
        handoff_df,
        ["source_id", "elite_id", "tf", "best_start", "best_end", "best_strand", "best_score_norm"],
        context="handoff_df",
    )
    if not pwms_by_source:
        raise ValueError("Missing source PWM set mapping for portfolio elite showcase.")
    if int(ncols) < 1:
        raise ValueError("Portfolio elite showcase ncols must be >= 1.")
    if int(dpi) < 72:
        raise ValueError("Portfolio elite showcase dpi must be >= 72.")

    showcase_cols = int(ncols)
    source_ids = selected_elites_df["source_id"].astype(str).drop_duplicates().tolist()
    records = []
    for source_id in source_ids:
        source_pwms = pwms_by_source.get(source_id)
        if source_pwms is None:
            raise ValueError(f"Missing source PWM set for portfolio elite showcase: source_id={source_id!r}")
        source_elites, source_hits, tf_names = _source_showcase_frames(
            selected_elites_df=selected_elites_df,
            handoff_df=handoff_df,
            source_id=source_id,
        )
        source_records = build_elites_showcase_records(
            elites_df=source_elites,
            hits_df=source_hits,
            tf_names=tf_names,
            pwms=source_pwms,
            max_panels=int(len(source_elites)),
            overlay_text_fn=lambda elite_row, tf_list: _portfolio_overlay_text(
                elite_row,
                tf_names=tf_list,
                ncols=showcase_cols,
            ),
            overlay_ncols=showcase_cols,
            meta_source="portfolio_elite_showcase",
        )
        records.extend(source_records)

    if not records:
        raise ValueError("Portfolio elite showcase produced zero records.")
    fig = render_record_grid_figure(
        records,
        ncols=showcase_cols,
        style_overrides=cruncher_showcase_style_overrides(),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=int(dpi), png_compress_level=9)
    plt.close(fig)
