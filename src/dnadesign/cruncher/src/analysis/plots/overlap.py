"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/overlap.py

Render motif placement and overlap diagnostics for selected elites.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from dnadesign.cruncher.analysis.plots._savefig import savefig
from dnadesign.cruncher.analysis.plots._style import apply_axes_style, place_figure_caption


def _required_columns(df: pd.DataFrame, columns: list[str], *, context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def _resolve_sequence_length(elites_df: pd.DataFrame, sequence_length: int | None) -> int:
    if isinstance(sequence_length, int) and sequence_length > 0:
        return int(sequence_length)
    if "sequence" not in elites_df.columns:
        raise ValueError("Elites table missing sequence column required to infer sequence length.")
    lengths = elites_df["sequence"].astype(str).str.len()
    if lengths.empty or lengths.min() <= 0:
        raise ValueError("Cannot infer sequence length from elites table.")
    if int(lengths.min()) != int(lengths.max()):
        raise ValueError("Elites sequences must share a single fixed length for placement plotting.")
    return int(lengths.iloc[0])


def _elite_order(elites_df: pd.DataFrame) -> list[str]:
    frame = elites_df.copy()
    frame["id"] = frame["id"].astype(str)
    if "rank" in frame.columns:
        frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
        frame = frame.sort_values(["rank", "id"], na_position="last")
    else:
        frame = frame.sort_values("id")
    return frame["id"].tolist()


def _hit_map(hits_df: pd.DataFrame, tf_names: list[str]) -> dict[tuple[str, str], tuple[int, int]]:
    _required_columns(hits_df, ["elite_id", "tf", "best_start", "pwm_width"], context="hits_df")
    map_by_key: dict[tuple[str, str], tuple[int, int]] = {}
    for elite_id, tf_name, start, width in hits_df[["elite_id", "tf", "best_start", "pwm_width"]].itertuples(
        index=False, name=None
    ):
        tf = str(tf_name)
        if tf not in tf_names:
            continue
        if not isinstance(start, (int, float)) or not isinstance(width, (int, float)):
            continue
        begin = int(start)
        end = int(start) + int(width)
        map_by_key[(str(elite_id), tf)] = (begin, end)
    return map_by_key


def _plot_tracks(
    ax: plt.Axes,
    *,
    elite_ids: list[str],
    tf_names: list[str],
    hit_by_key: dict[tuple[str, str], tuple[int, int]],
    sequence_length: int,
) -> dict[str, str]:
    cmap = plt.get_cmap("tab10", max(1, len(tf_names)))
    tf_colors = {tf: cmap(idx % cmap.N) for idx, tf in enumerate(tf_names)}
    n_elites = len(elite_ids)
    y_positions = {elite_id: n_elites - idx for idx, elite_id in enumerate(elite_ids)}

    for elite_id in elite_ids:
        y = y_positions[elite_id]
        ax.hlines(y, 0, sequence_length, color="#d0d0d0", linewidth=0.8, zorder=1)
        for tf in tf_names:
            span = hit_by_key.get((elite_id, tf))
            if span is None:
                continue
            start, end = span
            width = max(0, end - start)
            if width <= 0:
                continue
            patch = Rectangle(
                (start, y - 0.35),
                width,
                0.7,
                facecolor=tf_colors[tf],
                edgecolor="#111111",
                linewidth=0.5,
            )
            ax.add_patch(patch)

    ax.set_xlim(0, sequence_length)
    ax.set_ylim(0.2, n_elites + 0.8)
    ax.set_yticks([y_positions[elite_id] for elite_id in elite_ids])
    ax.set_yticklabels([f"elite {idx + 1}" for idx in range(len(elite_ids))], fontsize=8)
    ax.set_xlabel("Sequence position (bp)")
    ax.set_ylabel("Elite")
    ax.set_title("Best-hit motif windows per elite")
    apply_axes_style(ax, ygrid=False)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=tf_colors[tf], edgecolor="#111111", linewidth=0.5, label=tf)
        for tf in tf_names
    ]
    if handles:
        ax.legend(handles=handles, frameon=False, ncol=min(4, len(handles)), fontsize=8, loc="upper right")
    return {tf: str(tf_colors[tf]) for tf in tf_names}


def _plot_pairwise_scatter(
    ax: plt.Axes,
    *,
    elite_ids: list[str],
    hit_by_key: dict[tuple[str, str], tuple[int, int]],
    focus_pair: tuple[str, str] | None,
) -> None:
    if focus_pair is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "Pairwise placement unavailable", ha="center", va="center", fontsize=9)
        return
    tf_a, tf_b = focus_pair
    rows: list[tuple[str, int, int]] = []
    for elite_id in elite_ids:
        hit_a = hit_by_key.get((elite_id, tf_a))
        hit_b = hit_by_key.get((elite_id, tf_b))
        if hit_a is None or hit_b is None:
            continue
        rows.append((elite_id, int(hit_a[0]), int(hit_b[0])))
    if not rows:
        ax.axis("off")
        ax.text(0.5, 0.5, f"No paired best-hit positions for {tf_a} vs {tf_b}", ha="center", va="center", fontsize=9)
        return
    x_vals = np.asarray([row[1] for row in rows], dtype=float)
    y_vals = np.asarray([row[2] for row in rows], dtype=float)
    ax.scatter(x_vals, y_vals, s=34, c="#4c78a8", edgecolors="#111111", linewidths=0.6)
    low = float(min(np.min(x_vals), np.min(y_vals)))
    high = float(max(np.max(x_vals), np.max(y_vals)))
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1.0, color="#8f8f8f")
    for idx, (_, x_val, y_val) in enumerate(rows, start=1):
        ax.annotate(str(idx), xy=(x_val, y_val), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel(f"{tf_a} best-hit start")
    ax.set_ylabel(f"{tf_b} best-hit start")
    ax.set_title("Pairwise best-hit placement")
    apply_axes_style(ax, ygrid=True)


def _plot_overlap_summary(
    ax: plt.Axes,
    *,
    summary_df: pd.DataFrame,
    elite_overlap_df: pd.DataFrame,
    tf_names: list[str],
) -> None:
    if len(tf_names) > 2:
        matrix = pd.DataFrame(0.0, index=tf_names, columns=tf_names, dtype=float)
        for tf_i, tf_j, overlap_rate in summary_df[["tf_i", "tf_j", "overlap_rate"]].itertuples(index=False, name=None):
            if tf_i not in matrix.index or tf_j not in matrix.columns:
                continue
            value = float(overlap_rate) if isinstance(overlap_rate, (int, float)) else 0.0
            matrix.loc[tf_i, tf_j] = value
            matrix.loc[tf_j, tf_i] = value
        image = ax.imshow(matrix.to_numpy(dtype=float), vmin=0.0, vmax=1.0, cmap="mako", aspect="auto")
        ax.set_xticks(np.arange(len(tf_names)))
        ax.set_yticks(np.arange(len(tf_names)))
        ax.set_xticklabels(tf_names, fontsize=7, rotation=45, ha="right")
        ax.set_yticklabels(tf_names, fontsize=7)
        ax.set_title("Pairwise overlap fraction")
        apply_axes_style(ax, ygrid=False)
        cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Overlap fraction", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        for i, tf_i in enumerate(tf_names):
            for j, tf_j in enumerate(tf_names):
                ax.text(j, i, f"{matrix.loc[tf_i, tf_j]:.2f}", ha="center", va="center", fontsize=6, color="white")
        return

    _required_columns(elite_overlap_df, ["overlap_total_bp"], context="elite_overlap_df")
    overlap_vals = pd.to_numeric(elite_overlap_df["overlap_total_bp"], errors="coerce").dropna()
    if overlap_vals.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "Overlap summary unavailable", ha="center", va="center", fontsize=9)
        return
    y_vals = np.arange(len(overlap_vals), 0, -1)
    ax.scatter(overlap_vals.to_numpy(dtype=float), y_vals, s=34, c="#f58518", edgecolors="#111111", linewidths=0.6)
    ax.set_xlim(left=0.0)
    ax.set_xlabel("Overlap length (bp)")
    ax.set_ylabel("Elite index")
    ax.set_title("Overlap length across elites")
    apply_axes_style(ax, ygrid=True)


def plot_overlap_panel(
    summary_df: pd.DataFrame,
    elite_overlap_df: pd.DataFrame,
    tf_names: Iterable[str],
    out_path: Path,
    *,
    hits_df: pd.DataFrame,
    elites_df: pd.DataFrame,
    sequence_length: int | None = None,
    focus_pair: tuple[str, str] | None = None,
    dpi: int,
    png_compress_level: int,
) -> None:
    tf_list = [str(tf) for tf in tf_names]
    if summary_df is None or summary_df.empty or not tf_list:
        raise ValueError("Overlap summary is required for overlap panel.")
    if elite_overlap_df is None or elite_overlap_df.empty:
        raise ValueError("Elite overlap table is required for overlap panel.")
    if hits_df is None or hits_df.empty:
        raise ValueError("Hits table is required for motif placement panel.")
    if elites_df is None or elites_df.empty:
        raise ValueError("Elites table is required for motif placement panel.")
    _required_columns(elites_df, ["id"], context="elites_df")

    seq_len = _resolve_sequence_length(elites_df, sequence_length)
    elite_ids = _elite_order(elites_df)
    hit_by_key = _hit_map(hits_df, tf_list)
    selected_focus_pair = focus_pair
    if selected_focus_pair is None and len(tf_list) >= 2:
        selected_focus_pair = (tf_list[0], tf_list[1])

    fig = plt.figure(figsize=(12.0, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.28, wspace=0.22)
    ax_tracks = fig.add_subplot(grid[0, :])
    ax_pair = fig.add_subplot(grid[1, 0])
    ax_overlap = fig.add_subplot(grid[1, 1])

    _plot_tracks(
        ax_tracks,
        elite_ids=elite_ids,
        tf_names=tf_list,
        hit_by_key=hit_by_key,
        sequence_length=seq_len,
    )
    _plot_pairwise_scatter(
        ax_pair,
        elite_ids=elite_ids,
        hit_by_key=hit_by_key,
        focus_pair=selected_focus_pair,
    )
    _plot_overlap_summary(
        ax_overlap,
        summary_df=summary_df,
        elite_overlap_df=elite_overlap_df,
        tf_names=tf_list,
    )
    place_figure_caption(fig, "Best-hit windows are derived from stored elites_hits artifacts.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
