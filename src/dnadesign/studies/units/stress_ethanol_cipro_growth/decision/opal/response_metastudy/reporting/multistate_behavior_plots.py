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

_VIEWS = ("ethanol", "ciprofloxacin", "and")
_COLORS = {"ethanol": "#C97A20", "ciprofloxacin": "#3B78B5", "and": "#6C5AA7"}


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
    scenarios = _scenario_order(frame)
    holdout_index = 0
    scenario_labels: list[str] = []
    for scenario in scenarios:
        if scenario.startswith("quantile_"):
            scenario_labels.append(scenario.removeprefix("quantile_"))
        else:
            holdout_index += 1
            scenario_labels.append(f"E{holdout_index}")
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharey=True, constrained_layout=True)
    for axis, view in zip(axes, _VIEWS, strict=True):
        rows = frame.loc[frame["selection_view_id"].astype(str).eq(view)].set_index("scenario_id").loc[scenarios]
        x = np.arange(len(rows))
        axis.plot(x, rows["score_spearman_vs_primary"], "o-", color=_COLORS[view], label="Rank correlation")
        axis.plot(
            x,
            rows["raw_top_k_overlap"].to_numpy(dtype=float) / rows["raw_top_k"].to_numpy(dtype=float),
            "s--",
            color="#606770",
            label="Top-6 overlap fraction",
        )
        axis.set_title(_pretty_view(view), fontsize=12, fontweight="semibold")
        axis.set_xticks(x, scenario_labels, fontsize=8)
        axis.axvline(4.5, color="#B4B9BF", linewidth=0.8, linestyle=":")
        axis.set_ylim(-0.05, 1.05)
        _style_axis(axis, grid_axis="y")
    axes[0].set_ylabel("Agreement with primary q90", fontsize=11)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    figure.suptitle("Normalization robustness", fontsize=14, fontweight="semibold")
    figure.supxlabel("Scale quantile; E1–E8 are chronological Reader-experiment holdouts", fontsize=10)
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)


def _plot_grouped_validation(frame: pd.DataFrame, path: Path) -> None:
    summary = frame.drop_duplicates(["seed", "selection_view_id", "objective_name"])
    objectives = tuple(sorted(summary["objective_name"].astype(str).unique()))
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), sharey=True, constrained_layout=True)
    for axis, view in zip(axes, _VIEWS, strict=True):
        rows = summary.loc[summary["selection_view_id"].astype(str).eq(view)]
        for index, objective in enumerate(objectives):
            objective_rows = rows.loc[rows["objective_name"].astype(str).eq(objective)]
            within = objective_rows["median_within_group_spearman"].to_numpy(dtype=float)
            pooled = objective_rows["pooled_oof_spearman"].to_numpy(dtype=float)
            jitter = np.linspace(-0.08, 0.08, len(within))
            axis.scatter(index - 0.11 + jitter, within, facecolors="white", edgecolors=_COLORS[view], s=42, zorder=3)
            axis.scatter(index + 0.11 + jitter, pooled, color="#606770", marker="s", s=31, zorder=3)
            axis.hlines(np.median(within), index - 0.28, index - 0.02, color=_COLORS[view], linewidth=2.2)
            axis.hlines(np.median(pooled), index + 0.02, index + 0.28, color="#606770", linewidth=2.2)
        axis.axhline(0.0, color="#90959B", linewidth=1.0)
        axis.set_xticks(range(len(objectives)), [_objective_label(value) for value in objectives], fontsize=9)
        axis.set_title(_pretty_view(view), fontsize=12, fontweight="semibold")
        _style_axis(axis, grid_axis="y")
    axes[0].set_ylabel("Held-out Spearman correlation", fontsize=11)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].scatter([], [], facecolors="white", edgecolors="#555", s=42, label="Median within group")
    axes[0].scatter([], [], color="#606770", marker="s", s=31, label="Pooled out of fold")
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    figure.suptitle("Grouped prediction-to-truth validation", fontsize=14, fontweight="semibold")
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)


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
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 5.1))
    for axis, view in zip(axes, _VIEWS, strict=True):
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
            f"r{int(rank)} {str(label)[:12]} · {_coordinate_label(str(limit))}"
            for rank, label, limit in zip(rows["rank"], rows["display_label"], rows["limiting_coordinate"], strict=True)
        ]
        axis.set_yticks(y, row_labels, fontsize=8.5)
        axis.invert_yaxis()
        axis.set_title(_pretty_view(view), fontsize=12, fontweight="semibold")
        _style_axis(axis, grid_axis="x")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncols=5,
        frameon=False,
        fontsize=9,
    )
    figure.text(
        0.5,
        0.035,
        "Normalized score in assay-resolution units · zero is a reference direction, not a feasibility boundary",
        ha="center",
        fontsize=10,
        color="#555B63",
    )
    figure.suptitle(
        "Behavior allocation preview: family decomposition",
        y=0.985,
        fontsize=14,
        fontweight="semibold",
    )
    figure.tight_layout(rect=(0.01, 0.09, 0.99, 0.82), w_pad=4.5)
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)


def _style_axis(axis: plt.Axes, *, grid_axis: str) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=9, width=0.8)
    axis.grid(axis=grid_axis, color="#D9DDE2", linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)


def _scenario_order(frame: pd.DataFrame) -> list[str]:
    quantiles = sorted(frame.loc[frame["scenario_kind"].eq("scale_quantile"), "scenario_id"].astype(str).unique())
    holdouts = sorted(
        frame.loc[frame["scenario_kind"].eq("leave_one_source_experiment_out"), "scenario_id"].astype(str).unique()
    )
    return [*quantiles, *holdouts]


def _pretty_view(value: str) -> str:
    return "AND" if value == "and" else value.capitalize()


def _objective_label(value: str) -> str:
    return "Behavior" if value == "multistate_response_behavior_v1" else "RMF"


def _coordinate_label(value: str) -> str:
    family, state = value.split(":", maxsplit=1)
    state = state.translate(str.maketrans("01>", "₀₁›"))
    if family == "response":
        return f"Δr {state}"
    return f"b {state}" if family == "on_signal" else f"−b {state}"


__all__ = ["render_multistate_behavior_plots"]
