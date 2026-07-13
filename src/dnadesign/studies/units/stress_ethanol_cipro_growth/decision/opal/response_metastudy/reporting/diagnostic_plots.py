"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/diagnostic_plots.py

Metric diagnostic plots for the response metric metastudy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..core.contracts import RecommendationThresholds, StressTargetView
from .plot_helpers import focus_policy_ids, target_view_mask_map
from .plot_style import save_metastudy_figure
from .plot_vocabulary import compact_policy_label, policy_label, target_view_label

_SFXI_STATE_LABELS = ["v00\nNo stress", "v10\nEthanol", "v01\nCiprofloxacin", "v11\nBoth stresses"]


def write_selected_setpoint_residuals(
    summary: pd.DataFrame,
    candidates: pd.DataFrame,
    path: Path,
    *,
    comparison_policy_id: str,
    target_views: tuple[StressTargetView, ...],
) -> None:
    focus = focus_policy_ids(summary, comparison_policy_id=comparison_policy_id)[:2]
    setpoints = target_view_mask_map(target_views)
    rows: list[dict[str, object]] = []
    for (policy_id, selection_view_id), group in candidates[candidates["policy_id"].isin(focus)].groupby(
        ["policy_id", "selection_view_id"],
        sort=False,
    ):
        setpoint = setpoints[str(selection_view_id)]
        for idx, state in enumerate(("v00", "v10", "v01", "v11")):
            rows.append(
                {
                    "policy_id": str(policy_id),
                    "selection_view_id": str(selection_view_id),
                    "state": state,
                    "residual": float(group[state].mean()) - float(setpoint[idx]),
                }
            )
    data = pd.DataFrame(rows)
    _write_policy_target_view_heatmaps(
        data,
        path,
        policy_ids=focus,
        value_column="residual",
        cmap="vlag",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        colorbar_label="Mean selected SFXI logic - target setpoint",
    )


def write_logic_gate_feasibility(
    summary: pd.DataFrame,
    path: Path,
    *,
    thresholds: RecommendationThresholds,
) -> None:
    data = summary[summary["kind"] == "logic_gate"].copy().sort_values("logic_gate")
    data["Selected logic fidelity"] = data["min_target_view_median_logic"]
    data["Eligible candidates"] = data["min_eligible_count"]
    fig = plt.figure(figsize=(7.8, 4.6))
    ax = sns.scatterplot(
        data=data,
        x="logic_gate",
        y="min_effective_topk",
        hue="Selected logic fidelity",
        size="Eligible candidates",
        sizes=(60, 220),
        palette="crest",
        edgecolor="#333333",
        linewidth=0.4,
    )
    ax.axhline(thresholds.min_effective_topk, color="#4f8a5b", linestyle="--", linewidth=1.0)
    ax.axvline(thresholds.min_target_view_median_logic, color="#b94a48", linestyle="--", linewidth=1.0)
    ax.set_ylim(-0.3, thresholds.min_effective_topk + 0.7)
    ax.set_xlabel("Minimum logic-fidelity gate")
    ax.set_ylabel("Smallest target-view top-k")
    ax.set_title("Logic gate feasibility")
    ax.set_box_aspect(1)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    plt.tight_layout()
    save_metastudy_figure(fig, path)


def write_logic_effect_scatter(
    summary: pd.DataFrame,
    candidates: pd.DataFrame,
    path: Path,
    *,
    comparison_policy_id: str,
) -> None:
    focus = focus_policy_ids(summary, comparison_policy_id=comparison_policy_id)[:2]
    data = candidates[candidates["policy_id"].isin(focus)].copy()
    data["Target view"] = data["selection_view_id"].map(target_view_label)
    data["Policy"] = data["policy_id"].map(policy_label)
    grid = sns.relplot(
        data=data,
        x="logic_fidelity",
        y="effect_scaled",
        col="Target view",
        hue="Policy",
        kind="scatter",
        height=3.2,
        aspect=1.0,
        s=55,
    )
    grid.set_axis_labels("SFXI logic fidelity", "SFXI scaled effect")
    grid.set_titles("{col_name}")
    for axis in grid.axes.flat:
        axis.set_box_aspect(1)
    sns.move_legend(
        grid,
        "center left",
        bbox_to_anchor=(1.0, 0.5),
        title="Scoring policy",
        frameon=False,
    )
    grid.fig.suptitle("Top-k candidates: target fidelity versus effect", y=1.05)
    grid.tight_layout()
    save_metastudy_figure(grid.fig, path)


def write_score_correlation_matrix(
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    path: Path,
    *,
    comparison_policy_id: str,
) -> None:
    focus = focus_policy_ids(summary, comparison_policy_id=comparison_policy_id)
    data = pairwise[(pairwise["policy_id"].isin(focus)) & (pairwise["metric"] == "between_selection_views")].copy()
    data["pair"] = [
        _target_view_pair_label(selection_view_a, selection_view_b)
        for selection_view_a, selection_view_b in zip(
            data["selection_view_a"],
            data["selection_view_b"],
            strict=True,
        )
    ]
    data["policy"] = data["policy_id"].map(compact_policy_label)
    pivot = data.pivot_table(index="policy", columns="pair", values="pearson", aggfunc="first")
    fig = plt.figure(figsize=(7.8, 4.2))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="mako",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Score Pearson correlation"},
        square=True,
    )
    plt.title("Campaign score coupling")
    plt.xlabel("Campaign pair")
    plt.ylabel("Scoring policy")
    plt.xticks(rotation=0, fontsize=8)
    plt.tight_layout()
    save_metastudy_figure(fig, path)


def write_selected_vec8_profiles(
    summary: pd.DataFrame,
    candidates: pd.DataFrame,
    path: Path,
    *,
    comparison_policy_id: str,
) -> None:
    focus = focus_policy_ids(summary, comparison_policy_id=comparison_policy_id)[:2]
    data = candidates[candidates["policy_id"].isin(focus)].copy()
    mean = (
        data.groupby(["policy_id", "selection_view_id"])[["v00", "v10", "v01", "v11"]]
        .mean()
        .reset_index()
        .melt(id_vars=["policy_id", "selection_view_id"], var_name="state", value_name="predicted_logic")
    )
    _write_policy_target_view_heatmaps(
        mean,
        path,
        policy_ids=focus,
        value_column="predicted_logic",
        cmap="crest",
        vmin=0.0,
        vmax=1.0,
        center=None,
        colorbar_label="Mean predicted SFXI logic",
    )


def _write_policy_target_view_heatmaps(
    data: pd.DataFrame,
    path: Path,
    *,
    policy_ids: list[str],
    value_column: str,
    cmap: str,
    vmin: float,
    vmax: float,
    center: float | None,
    colorbar_label: str,
) -> None:
    figure = plt.figure(figsize=(9.8, 4.5), constrained_layout=True)
    layout = figure.add_gridspec(1, len(policy_ids) + 1, width_ratios=[1.0] * len(policy_ids) + [0.06])
    axes = [figure.add_subplot(layout[0, index]) for index in range(len(policy_ids))]
    colorbar_axis = figure.add_subplot(layout[0, -1])
    for index, (axis, policy_id) in enumerate(zip(axes, policy_ids, strict=True)):
        selected = data.loc[data["policy_id"].astype(str).eq(policy_id)].copy()
        if selected.empty:
            raise ValueError(f"policy heatmap has no rows for {policy_id!r}.")
        pivot = selected.pivot_table(
            index="selection_view_id",
            columns="state",
            values=value_column,
            aggfunc="first",
        )
        pivot = pivot.reindex(index=[value for value in ("ethanol", "ciprofloxacin", "and") if value in pivot.index])
        pivot = pivot[["v00", "v10", "v01", "v11"]]
        pivot.index = [target_view_label(value) for value in pivot.index]
        pivot.columns = _SFXI_STATE_LABELS
        is_last = index == len(policy_ids) - 1
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            center=center,
            vmin=vmin,
            vmax=vmax,
            cbar=is_last,
            cbar_ax=colorbar_axis if is_last else None,
            cbar_kws={"label": colorbar_label},
            square=True,
            ax=axis,
        )
        axis.set_title(policy_label(policy_id))
        axis.set_xlabel("SFXI logic state")
        axis.set_ylabel("Target view" if index == 0 else "")
        axis.tick_params(axis="x", rotation=0, labelsize=8)
        axis.tick_params(axis="y", rotation=0, labelsize=9)
    save_metastudy_figure(figure, path)


def _target_view_pair_label(selection_view_a: object, selection_view_b: object) -> str:
    first = target_view_label(selection_view_a)
    second = target_view_label(selection_view_b)
    if first == "Ethanol" and second == "Ciprofloxacin":
        return "Ethanol vs\nCiprofloxacin"
    return f"{first}\nvs {second}"
