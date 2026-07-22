"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/metric_behavior_plots.py

Metric-behavior plots for the response metric metastudy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from .plot_style import save_metastudy_figure
from .plot_vocabulary import (
    compact_policy_label,
    panel_role_label,
    prediction_component_label,
    target_view_label,
)


def write_denominator_sensitivity(denominator_sensitivity: pd.DataFrame, path: Path) -> None:
    data = denominator_sensitivity.melt(
        id_vars=["policy_id", "selection_view_id", "denominator_factor"],
        value_vars=["median_logic_fidelity", "median_effect_scaled"],
        var_name="metric",
        value_name="value",
    )
    data["Target view"] = data["selection_view_id"].map(target_view_label)
    data["Policy"] = data["policy_id"].map(compact_policy_label)
    data["Summary component"] = data["metric"].map(prediction_component_label)
    grid = sns.relplot(
        data=data,
        x="denominator_factor",
        y="value",
        col="Target view",
        hue="Policy",
        style="Summary component",
        kind="line",
        marker="o",
        height=3.2,
        aspect=1.0,
    )
    grid.set_axis_labels("", "Top-K median")
    grid.set(ylim=(0.0, 1.0))
    grid.set_titles("{col_name}")
    for axis in grid.axes.flat:
        axis.set_box_aspect(1)
    grid.axes.flat[len(grid.axes.flat) // 2].set_xlabel("Denominator factor")
    grid.fig.set_layout_engine("constrained")
    sns.move_legend(grid, "outside lower center", ncol=3, frameon=False)
    grid.fig.suptitle("Denominator sensitivity", y=1.05)
    save_metastudy_figure(grid.fig, path)


def write_policy_comparison_panel_roles(comparison_panel: pd.DataFrame, path: Path) -> None:
    data = (
        comparison_panel.groupby(["panel_role", "selection_view_id"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["panel_role", "selection_view_id"], kind="mergesort")
    )
    data["Panel role"] = data["panel_role"].map(panel_role_label)
    data["Target view"] = data["selection_view_id"].map(target_view_label)
    fig = plt.figure(figsize=(9.4, max(4.4, 0.42 * data["panel_role"].nunique())))
    ax = sns.barplot(data=data, y="Panel role", x="rows", hue="Target view", orient="h")
    ax.set_title("Policy-comparison panel role coverage")
    ax.set_xlabel("Candidate-panel rows")
    ax.set_ylabel("Diagnostic stratum")
    ax.legend(fontsize=11, loc="best")
    plt.tight_layout()
    save_metastudy_figure(fig, path)
