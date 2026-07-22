"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/model_validation_plot.py

Held-out model validation plot for the response metric metastudy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .plot_style import save_metastudy_figure
from .plot_vocabulary import model_metric_label

_STRATEGY_ORDER = ("leave_one_experiment_out", "shuffled_kfold")
_STRATEGY_LABELS = {
    "leave_one_experiment_out": "Leave one Reader experiment out",
    "shuffled_kfold": "Shuffled five-fold",
}
_SCOPE_COLORS = {"target": "#2A6F97", "selection_view_objective": "#B23A48"}


def write_model_validation(frame: pd.DataFrame, path: Path) -> None:
    required = {"split_strategy", "scope", "metric_id", "spearman"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"model validation plot missing columns: {missing}")
    present_strategies = set(frame["split_strategy"].astype(str))
    unknown_strategies = sorted(present_strategies - set(_STRATEGY_ORDER))
    if unknown_strategies:
        raise ValueError(f"model validation plot has unregistered split strategies: {unknown_strategies}")
    strategies = [strategy for strategy in _STRATEGY_ORDER if strategy in present_strategies]
    fig, axes = plt.subplots(
        1,
        len(strategies),
        figsize=(6.2 * len(strategies), 6.2),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for ax, strategy in zip(axes[0], strategies, strict=True):
        selected = frame.loc[frame["split_strategy"].astype(str).eq(strategy)]
        summary = (
            selected.groupby(["scope", "metric_id"], sort=False)["spearman"].agg(["median", "min", "max"]).reset_index()
        )
        labels = [model_metric_label(value).replace("\n", ": ") for value in summary["metric_id"]]
        positions = np.arange(len(summary))
        try:
            colors = [_SCOPE_COLORS[str(scope)] for scope in summary["scope"]]
        except KeyError as exc:
            raise ValueError(f"model validation plot has unregistered scope: {exc.args[0]!r}") from exc
        lower = summary["median"].to_numpy() - summary["min"].to_numpy()
        upper = summary["max"].to_numpy() - summary["median"].to_numpy()
        ax.errorbar(
            summary["median"],
            positions,
            xerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor="#6B7280",
            capsize=3,
            linewidth=1.2,
        )
        ax.scatter(summary["median"], positions, c=colors, s=54, zorder=3)
        ax.axvline(0.0, color="#111827", linewidth=0.9)
        ax.set_yticks(positions, labels, fontsize=11)
        ax.set_xlabel("Held-out Spearman correlation")
        ax.set_title(_STRATEGY_LABELS[strategy])
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_ylabel("Predicted response target")
    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color=color, label=label)
            for label, color in (
                ("Vec8 target", _SCOPE_COLORS["target"]),
                ("Selection-view objective", _SCOPE_COLORS["selection_view_objective"]),
            )
        ],
        loc="outside lower center",
        ncols=2,
        frameon=False,
    )
    save_metastudy_figure(fig, path)
