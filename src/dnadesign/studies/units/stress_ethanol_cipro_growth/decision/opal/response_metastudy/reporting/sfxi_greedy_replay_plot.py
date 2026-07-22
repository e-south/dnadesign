"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/sfxi_greedy_replay_plot.py

Publication plot for the persisted historical SFXI greedy selections.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .plot_style import save_metastudy_figure

_VIEW_ORDER = ("ethanol", "ciprofloxacin", "and")
_VIEW_TITLES = {
    "ethanol": "Ethanol-associated",
    "ciprofloxacin": "Ciprofloxacin-associated",
    "and": "Combined-state-only",
}
_POOL_COLOR = "#d1d5db"
_VIEW_SPECIFIC_COLOR = "#0f766e"
_SHARED_COLOR = "#d97706"


def write_historical_sfxi_greedy_replay(
    canonical_scored: dict[str, pd.DataFrame],
    replay: pd.DataFrame,
    path: Path,
) -> None:
    """Show the exact persisted Top-K against each complete prediction surface."""

    _validate_inputs(canonical_scored, replay)
    x_limits, y_limits = _shared_limits(canonical_scored)
    pool_count = len(canonical_scored[_VIEW_ORDER[0]])
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 5.0), sharex=True, sharey=True, layout="constrained")
    layout_engine = figure.get_layout_engine()
    if layout_engine is None:
        raise RuntimeError("SFXI greedy replay requires Matplotlib constrained layout.")
    layout_engine.set(rect=(0.02, 0.03, 0.96, 0.84), w_pad=0.04, h_pad=0.03, wspace=0.05)
    for axis, view_id in zip(axes, _VIEW_ORDER, strict=True):
        pool = canonical_scored[view_id]
        selected = replay.loc[replay["selection_view_id"] == view_id].sort_values("rank", kind="mergesort")
        axis.hexbin(
            pool["logic_fidelity"].to_numpy(dtype=float),
            pool["effect_scaled"].to_numpy(dtype=float),
            gridsize=72,
            mincnt=1,
            bins="log",
            cmap="Greys",
            linewidths=0.0,
            alpha=0.78,
            extent=(*x_limits, *y_limits),
            zorder=1,
            rasterized=True,
        )
        colors = np.where(selected["selection_view_count"].to_numpy(dtype=int) > 1, _SHARED_COLOR, _VIEW_SPECIFIC_COLOR)
        axis.scatter(
            selected["logic_fidelity"],
            selected["effect_scaled"],
            s=104,
            c=colors,
            edgecolors="#111827",
            linewidths=1.15,
            zorder=5,
        )
        _annotate_ranks(axis, selected, x_limits=x_limits, y_limits=y_limits)
        summary = selected.iloc[0]
        axis.text(
            0.5,
            1.025,
            "Rank agreement with SFXI\n"
            f"Scaled effect: ρ = {float(summary['score_vs_effect_spearman']):.2f} · "
            f"Logic fidelity: ρ = {float(summary['score_vs_logic_spearman']):.2f}",
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            color="#111827",
            linespacing=1.08,
            zorder=7,
        )
        axis.set_title(_VIEW_TITLES[view_id], fontsize=15, pad=40)
        axis.set_xlabel(r"Logic fidelity, $F_{\mathrm{logic}}$", fontsize=13)
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
        axis.set_box_aspect(1)
        axis.tick_params(labelsize=11)

    first = replay.iloc[0]
    axes[0].set_ylabel(r"Scaled effect, $E_{\mathrm{scaled}}$", fontsize=13)
    figure.suptitle("SFXI greedy selection replay", fontsize=18, fontweight="semibold", y=0.995)
    figure.legend(
        handles=[
            Patch(facecolor=_POOL_COLOR, edgecolor="none", label="All eligible predictions (density)"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=_VIEW_SPECIFIC_COLOR,
                markeredgecolor="#111827",
                markersize=8,
                label="Selected in one view",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=_SHARED_COLOR,
                markeredgecolor="#111827",
                markersize=8,
                label="Selected in multiple views",
            ),
        ],
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=11,
        handletextpad=0.55,
        columnspacing=1.25,
    )
    figure.text(
        0.5,
        0.925,
        f"{pool_count:,} eligible predictions per view; exact Top-6 numbered · "
        f"{int(first['total_selection_slots'])} view slots → "
        f"{int(first['unique_selected_sequences'])} unique sequences; "
        f"{int(first['selected_in_all_views'])} selected in all three views.",
        ha="center",
        va="center",
        fontsize=12,
        color="#111827",
    )
    save_metastudy_figure(figure, path)


def _annotate_ranks(
    axis: plt.Axes,
    selected: pd.DataFrame,
    *,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    x_span = x_limits[1] - x_limits[0]
    y_span = y_limits[1] - y_limits[0]
    work = selected.sort_values(["effect_scaled", "rank"], ascending=[False, True], kind="mergesort").copy()
    label_x = float(work["logic_fidelity"].max()) + 0.065 * x_span
    place_left = label_x > x_limits[1] - 0.035 * x_span
    if place_left:
        label_x = float(work["logic_fidelity"].min()) - 0.065 * x_span
    spacing = max(0.075 * y_span, float(work["effect_scaled"].max() - work["effect_scaled"].min()) / 5.0)
    label_top = min(float(work["effect_scaled"].max()), y_limits[1] - 0.055 * y_span)
    label_bottom = label_top - spacing * (len(work) - 1)
    lower_bound = y_limits[0] + 0.055 * y_span
    if label_bottom < lower_bound:
        label_top += lower_bound - label_bottom
    for label_y, row in zip(
        np.linspace(label_top, label_top - spacing * (len(work) - 1), len(work)),
        work.itertuples(index=False),
        strict=True,
    ):
        axis.annotate(
            str(int(row.rank)),
            (float(row.logic_fidelity), float(row.effect_scaled)),
            xytext=(label_x, float(label_y)),
            textcoords="data",
            ha="right" if place_left else "left",
            va="center",
            fontsize=11,
            fontweight="semibold",
            color="#111827",
            arrowprops={"arrowstyle": "-", "color": "#6b7280", "linewidth": 0.8, "shrinkA": 2, "shrinkB": 5},
            zorder=7,
        )


def _shared_limits(canonical_scored: dict[str, pd.DataFrame]) -> tuple[tuple[float, float], tuple[float, float]]:
    all_logic = np.concatenate(
        [canonical_scored[view_id]["logic_fidelity"].to_numpy(dtype=float) for view_id in _VIEW_ORDER]
    )
    all_effect = np.concatenate(
        [canonical_scored[view_id]["effect_scaled"].to_numpy(dtype=float) for view_id in _VIEW_ORDER]
    )
    return _padded_limits(all_logic, floor=0.0, ceiling=1.0), _padded_limits(all_effect, floor=0.0, ceiling=1.0)


def _padded_limits(values: np.ndarray, *, floor: float, ceiling: float) -> tuple[float, float]:
    low = float(np.min(values))
    high = float(np.max(values))
    span = max(high - low, 0.05)
    return max(floor, low - 0.06 * span), min(ceiling, high + 0.09 * span)


def _validate_inputs(canonical_scored: dict[str, pd.DataFrame], replay: pd.DataFrame) -> None:
    if set(canonical_scored) != set(_VIEW_ORDER):
        raise ValueError(f"historical SFXI greedy plot requires views {_VIEW_ORDER}.")
    required_pool = {"id", "logic_fidelity", "effect_scaled", "score"}
    for view_id, frame in canonical_scored.items():
        missing = sorted(required_pool - set(frame.columns))
        if missing or frame.empty:
            raise ValueError(f"{view_id}: historical SFXI greedy plot pool is invalid; missing={missing}.")
    pool_counts = {len(frame) for frame in canonical_scored.values()}
    if len(pool_counts) != 1:
        raise ValueError("SFXI greedy plot requires one common prediction-pool size across target views.")
    required_replay = {
        "selection_view_id",
        "rank",
        "id",
        "logic_fidelity",
        "effect_scaled",
        "selection_view_count",
        "score_vs_effect_spearman",
        "score_vs_logic_spearman",
        "total_selection_slots",
        "unique_selected_sequences",
        "selected_in_all_views",
    }
    missing = sorted(required_replay - set(replay.columns))
    if missing or replay.empty:
        raise ValueError(f"historical SFXI greedy plot replay is invalid; missing={missing}.")
    if set(replay["selection_view_id"].astype(str)) != set(_VIEW_ORDER):
        raise ValueError("historical SFXI greedy plot replay does not contain all three target views.")
    counts = replay.groupby("selection_view_id")["rank"].size()
    if counts.nunique() != 1:
        raise ValueError("historical SFXI greedy plot requires the same Top-K in every target view.")


__all__ = ["write_historical_sfxi_greedy_replay"]
