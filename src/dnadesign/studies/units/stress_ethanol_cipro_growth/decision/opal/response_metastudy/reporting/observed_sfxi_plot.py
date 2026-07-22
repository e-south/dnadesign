"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/observed_sfxi_plot.py

Measured-label SFXI decomposition figure for the historical comparator.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from .plot_style import save_metastudy_figure

_VIEW_ORDER = ("ethanol", "ciprofloxacin", "and")
_VIEW_LABELS = {
    "ethanol": "Ethanol-associated",
    "ciprofloxacin": "Ciprofloxacin-associated",
    "and": "Combined-state-only",
}
_LIMITS = (-0.025, 1.025)
_TICKS = (0.0, 0.25, 0.5, 0.75, 1.0)


def write_historical_observed_sfxi_decomposition(
    components: pd.DataFrame,
    robustness: pd.DataFrame,
    path: Path,
) -> None:
    """Plot canonical SFXI components for the exact historical observed corpus."""

    _validate_inputs(components, robustness)
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.8), sharex=True, sharey=True, layout="constrained")
    layout_engine = fig.get_layout_engine()
    if layout_engine is None:
        raise RuntimeError("Observed SFXI decomposition requires Matplotlib constrained layout.")
    layout_engine.set(rect=(0.02, 0.03, 0.96, 0.86), w_pad=0.04, h_pad=0.03, wspace=0.05)
    fig.suptitle("Observed SFXI component decomposition", fontsize=18, fontweight="semibold", y=0.995)
    for axis, view_id in zip(axes, _VIEW_ORDER, strict=True):
        view = components.loc[components["selection_view_id"].astype(str).eq(view_id)].copy()
        summary = robustness.loc[
            robustness["selection_view_id"].astype(str).eq(view_id)
            & robustness["sensitivity_scope"].astype(str).eq("all_observed_labels")
        ]
        if len(summary) != 1:
            raise ValueError(f"Historical observed SFXI plot requires one full-corpus summary for {view_id!r}.")
        axis.scatter(
            view["logic_fidelity"],
            view["effect_scaled"],
            s=58,
            facecolor="#9ca3af",
            edgecolor="white",
            linewidth=0.8,
            alpha=0.92,
            zorder=3,
        )
        highest = view.loc[view["is_highest_observed_sfxi"].astype(bool)]
        axis.scatter(
            highest["logic_fidelity"],
            highest["effect_scaled"],
            s=108,
            facecolor="none",
            edgecolor="#111827",
            linewidth=1.6,
            zorder=4,
        )
        _annotate_controls(axis, view)
        row = summary.iloc[0]
        axis.text(
            0.5,
            1.025,
            "Rank agreement with SFXI\n"
            f"Scaled effect: $\\rho$ = {float(row['sfxi_vs_effect_spearman']):.2f} · "
            rf"Logic fidelity: $\rho$ = {float(row['sfxi_vs_logic_spearman']):.2f}",
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            linespacing=1.08,
            zorder=6,
        )
        axis.set_title(_VIEW_LABELS[view_id], fontsize=15, pad=40)
        axis.set_box_aspect(1)
        axis.set_xlim(_LIMITS)
        axis.set_ylim(_LIMITS)
        axis.set_xticks(_TICKS)
        axis.set_yticks(_TICKS)
        axis.tick_params(labelsize=11)
        axis.set_xlabel(r"Logic fidelity, $F_{\ell}$", fontsize=13)
    axes[0].set_ylabel(r"Scaled target-state effect, $E_{\mathrm{scaled}}$", fontsize=13)
    fig.legend(
        handles=_legend_handles(),
        loc="outside lower center",
        ncols=3,
        frameon=False,
        fontsize=11,
        handletextpad=0.6,
        columnspacing=1.7,
    )
    save_metastudy_figure(fig, path)


def _annotate_controls(axis: plt.Axes, view: pd.DataFrame) -> None:
    controls = view.loc[view["control_role"].astype(str).ne("")]
    if set(controls["control_role"].astype(str)) != {"SpyP", "sulAp"}:
        raise ValueError("Historical observed SFXI plot requires exactly the SpyP and sulAp controls.")
    vertical_offsets = {"SpyP": 13.0, "sulAp": -18.0}
    for row in controls.itertuples(index=False):
        x = float(row.logic_fidelity)
        y = float(row.effect_scaled)
        right_aligned = x >= 0.75
        axis.annotate(
            str(row.control_role),
            xy=(x, y),
            xytext=(-9 if right_aligned else 9, vertical_offsets[str(row.control_role)]),
            textcoords="offset points",
            ha="right" if right_aligned else "left",
            va="center",
            fontsize=10.5,
            color="#111827",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.95},
            arrowprops={"arrowstyle": "-", "color": "#6b7280", "linewidth": 0.8},
            annotation_clip=True,
            zorder=7,
        )


def _legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="#9ca3af",
            markeredgecolor="white",
            markersize=8,
            label="Observed SFXI label",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#111827",
            markeredgewidth=1.5,
            markersize=10,
            label="Highest six measured SFXI scores",
        ),
    ]


def _validate_inputs(components: pd.DataFrame, robustness: pd.DataFrame) -> None:
    required = {
        "id",
        "selection_view_id",
        "logic_fidelity",
        "effect_scaled",
        "sfxi",
        "is_highest_observed_sfxi",
        "control_role",
    }
    if missing := sorted(required - set(components.columns)):
        raise ValueError(f"Historical observed SFXI plot lacks component columns: {missing}")
    expected_rows = 35 * len(_VIEW_ORDER)
    if len(components) != expected_rows:
        raise ValueError(
            f"Historical observed SFXI plot expects {expected_rows} component rows; found {len(components)}."
        )
    counts = components.groupby("selection_view_id")["id"].nunique().to_dict()
    if counts != {view_id: 35 for view_id in _VIEW_ORDER}:
        raise ValueError(
            f"Historical observed SFXI plot requires 35 unique candidates per target view; found {counts}."
        )
    for view_id in _VIEW_ORDER:
        view = components.loc[components["selection_view_id"].astype(str).eq(view_id)]
        if int(view["is_highest_observed_sfxi"].astype(bool).sum()) != 6:
            raise ValueError(f"Historical observed SFXI plot requires six highlighted scores for {view_id!r}.")
    summary_required = {
        "selection_view_id",
        "sensitivity_scope",
        "sfxi_vs_logic_spearman",
        "sfxi_vs_effect_spearman",
    }
    if missing := sorted(summary_required - set(robustness.columns)):
        raise ValueError(f"Historical observed SFXI plot lacks robustness columns: {missing}")


__all__ = ["write_historical_observed_sfxi_decomposition"]
