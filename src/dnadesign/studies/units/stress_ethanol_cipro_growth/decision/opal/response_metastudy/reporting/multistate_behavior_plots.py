"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/multistate_behavior_plots.py

Minimal publication plots for the multistate behavior completion gate.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .multistate_behavior_plot_labels import (
    VIEW_COLORS,
    VIEW_ORDER,
    coordinate_label,
    objective_label,
    scenario_order,
    view_label,
)
from .multistate_behavior_plot_style import (
    AXIS_LABEL_SIZE,
    FIGURE_TITLE_SIZE,
    LEGEND_SIZE,
    PANEL_TITLE_SIZE,
    TICK_SIZE,
    save_figure,
    style_axis,
)


def render_multistate_behavior_plots(
    *,
    normalization_sensitivity: pd.DataFrame,
    grouped_validation: pd.DataFrame,
    allocation_comparison: pd.DataFrame,
    prediction_scores: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Render only the three plots needed to adjudicate the shadow objective."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    outputs = {
        "plot__normalization_robustness": root / "normalization_robustness.png",
        "plot__grouped_objective_validation": root / "grouped_objective_validation.png",
        "plot__allocation_family_decomposition": root / "allocation_family_decomposition.png",
    }
    _plot_normalization(normalization_sensitivity, outputs["plot__normalization_robustness"])
    _plot_grouped_validation(grouped_validation, outputs["plot__grouped_objective_validation"])
    _plot_allocation_families(
        allocation_comparison,
        prediction_scores,
        outputs["plot__allocation_family_decomposition"],
    )
    return outputs


def _plot_normalization(frame: pd.DataFrame, path: Path) -> None:
    scenarios = scenario_order(frame)
    holdout_index = 0
    scenario_labels: list[str] = []
    for scenario in scenarios:
        if scenario.startswith("quantile_"):
            scenario_labels.append(scenario.removeprefix("quantile_"))
        else:
            holdout_index += 1
            scenario_labels.append(f"E{holdout_index}")
    figure, axes = plt.subplots(1, 3, figsize=(15.6, 5.2), sharey=True, constrained_layout=True)
    for axis, view in zip(axes, VIEW_ORDER, strict=True):
        rows = frame.loc[frame["selection_view_id"].astype(str).eq(view)].set_index("scenario_id").loc[scenarios]
        x = np.arange(len(rows))
        axis.plot(x, rows["score_spearman_vs_primary"], "o-", color=VIEW_COLORS[view], label="Rank correlation")
        axis.plot(
            x,
            rows["raw_top_k_overlap"].to_numpy(dtype=float) / rows["raw_top_k"].to_numpy(dtype=float),
            "s--",
            color="#606770",
            label="Top-6 overlap fraction",
        )
        axis.set_title(view_label(view), fontsize=PANEL_TITLE_SIZE, fontweight="semibold")
        axis.set_xticks(x, scenario_labels, fontsize=TICK_SIZE)
        axis.axvline(4.5, color="#B4B9BF", linewidth=0.8, linestyle=":")
        axis.set_ylim(-0.05, 1.05)
        style_axis(axis, grid_axis="y")
    axes[0].set_ylabel("Agreement with primary q90", fontsize=AXIS_LABEL_SIZE)
    axes[0].legend(frameon=False, fontsize=LEGEND_SIZE, loc="lower left")
    figure.suptitle("Normalization robustness", fontsize=FIGURE_TITLE_SIZE, fontweight="semibold")
    figure.supxlabel(
        "Scale quantile; E1–E8 are chronological Reader-experiment holdouts",
        fontsize=AXIS_LABEL_SIZE,
    )
    save_figure(figure, path)


def _plot_grouped_validation(frame: pd.DataFrame, path: Path) -> None:
    summary = frame.drop_duplicates(["seed", "selection_view_id", "objective_name"])
    objectives = tuple(sorted(summary["objective_name"].astype(str).unique()))
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 5.4), sharey=True, constrained_layout=True)
    for axis, view in zip(axes, VIEW_ORDER, strict=True):
        rows = summary.loc[summary["selection_view_id"].astype(str).eq(view)]
        for index, objective in enumerate(objectives):
            objective_rows = rows.loc[rows["objective_name"].astype(str).eq(objective)]
            within = objective_rows["median_within_group_spearman"].to_numpy(dtype=float)
            pooled = objective_rows["pooled_oof_spearman"].to_numpy(dtype=float)
            jitter = np.linspace(-0.08, 0.08, len(within))
            axis.scatter(
                index - 0.11 + jitter,
                within,
                facecolors="white",
                edgecolors=VIEW_COLORS[view],
                s=42,
                zorder=3,
            )
            axis.scatter(index + 0.11 + jitter, pooled, color="#606770", marker="s", s=31, zorder=3)
            axis.hlines(np.median(within), index - 0.28, index - 0.02, color=VIEW_COLORS[view], linewidth=2.2)
            axis.hlines(np.median(pooled), index + 0.02, index + 0.28, color="#606770", linewidth=2.2)
        axis.axhline(0.0, color="#90959B", linewidth=1.0)
        axis.set_xticks(
            range(len(objectives)),
            [objective_label(value) for value in objectives],
            fontsize=TICK_SIZE,
        )
        axis.set_title(view_label(view), fontsize=PANEL_TITLE_SIZE, fontweight="semibold")
        style_axis(axis, grid_axis="y")
    axes[0].set_ylabel("Held-out Spearman correlation", fontsize=AXIS_LABEL_SIZE)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].scatter([], [], facecolors="white", edgecolors="#555", s=42, label="Median within group")
    axes[0].scatter([], [], color="#606770", marker="s", s=31, label="Pooled out of fold")
    axes[0].legend(frameon=False, fontsize=LEGEND_SIZE, loc="lower left")
    figure.suptitle(
        "Grouped prediction-to-truth validation",
        fontsize=FIGURE_TITLE_SIZE,
        fontweight="semibold",
    )
    save_figure(figure, path)


def _plot_allocation_families(allocation: pd.DataFrame, scores: pd.DataFrame, path: Path) -> None:
    selected = allocation.loc[allocation["objective_name"].astype(str).eq("multistate_response_behavior_v1")]
    columns = [
        "behavior_score",
        "hard_bottleneck_clearance",
        "response_family_score",
        "on_signal_family_score",
        "off_signal_suppression_family_score",
    ]
    labels = ["Behavior", "Hard bottleneck", "Response", "ON signal", "OFF signal suppression"]
    markers = ["D", "X", "o", "^", "s"]
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 6.2))
    for axis, view in zip(axes, VIEW_ORDER, strict=True):
        view_selected = selected.loc[selected["selection_view_id"].astype(str).eq(view)]
        rows = view_selected.merge(
            scores.loc[
                scores["selection_view_id"].astype(str).eq(view),
                ["id", "limiting_coordinate", *columns],
            ],
            on="id",
            how="left",
            validate="one_to_one",
        ).sort_values("allocation_slot", kind="mergesort")
        y = np.arange(len(rows))
        for column, label, marker in zip(columns, labels, markers, strict=True):
            axis.scatter(rows[column], y, s=44, marker=marker, label=label, zorder=3)
        axis.axvline(0.0, color="#90959B", linewidth=1.0)
        row_labels = [
            f"r{int(rank)} {str(label)[:12]} · {coordinate_label(str(limit))}"
            for rank, label, limit in zip(rows["rank"], rows["display_label"], rows["limiting_coordinate"], strict=True)
        ]
        axis.set_yticks(y, row_labels, fontsize=TICK_SIZE)
        axis.invert_yaxis()
        axis.set_title(view_label(view), fontsize=PANEL_TITLE_SIZE, fontweight="semibold")
        style_axis(axis, grid_axis="x")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncols=5,
        frameon=False,
        fontsize=LEGEND_SIZE,
    )
    figure.text(
        0.5,
        0.035,
        "Behavior score in log₂ units · zero is a reference direction, not a feasibility boundary",
        ha="center",
        fontsize=AXIS_LABEL_SIZE,
        color="#555B63",
    )
    figure.suptitle(
        "Behavior allocation preview: family decomposition",
        y=0.985,
        fontsize=FIGURE_TITLE_SIZE,
        fontweight="semibold",
    )
    figure.tight_layout(rect=(0.01, 0.09, 0.99, 0.86), w_pad=4.5)
    save_figure(figure, path)


__all__ = ["render_multistate_behavior_plots"]
