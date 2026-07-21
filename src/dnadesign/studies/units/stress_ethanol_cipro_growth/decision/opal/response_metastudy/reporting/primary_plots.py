"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/primary_plots.py

Primary decision plots for the response metric metastudy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..core.contracts import RecommendationThresholds
from .plot_helpers import focus_policy_ids
from .plot_style import save_metastudy_figure
from .plot_vocabulary import policy_label, target_view_label


def write_policy_guardrail_matrix(
    summary: pd.DataFrame,
    path: Path,
    *,
    thresholds: RecommendationThresholds,
    comparison_policy_id: str,
    model_support_passed: bool,
) -> None:
    data = summary.copy()
    focus_ids = set(focus_policy_ids(summary, comparison_policy_id=comparison_policy_id))
    data["is_focus"] = data["policy_id"].isin(focus_ids)
    data = data.sort_values(
        ["is_focus", "min_target_view_median_logic", "all_target_views_overlap", "policy_id"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).head(18)
    checks = (
        pd.DataFrame(
            {
                "Full top-k": (data["min_effective_topk"] >= thresholds.min_effective_topk).to_numpy(),
                "Eligible pool": (data["min_eligible_count"] >= thresholds.min_eligible_count).to_numpy(),
                "Logic fidelity": (
                    data["min_target_view_median_logic"] >= thresholds.min_target_view_median_logic
                ).to_numpy(),
                "Target-view\noverlap": (
                    data["all_target_views_overlap"] <= thresholds.max_all_target_views_overlap
                ).to_numpy(),
                "Score\ncoupling": (
                    data["mean_pairwise_score_spearman"] <= thresholds.max_mean_pairwise_score_spearman
                ).to_numpy(),
                "Held-out model": model_support_passed,
            },
            index=[fill(str(value), width=32) for value in data["label"]],
        )
        .fillna(False)
        .astype(int)
    )
    annotations = checks.replace({0: "fail", 1: "pass"})
    fig, ax = plt.subplots(figsize=(11.0, max(5.2, 0.46 * len(checks))))
    sns.heatmap(
        checks,
        annot=annotations,
        fmt="",
        cmap=sns.color_palette(["#e5e7eb", "#2563eb"], as_cmap=True),
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Promotion check")
    ax.set_ylabel("Scoring policy")
    fig.tight_layout()
    save_metastudy_figure(fig, path)


def write_policy_decision_frontier(
    summary: pd.DataFrame,
    path: Path,
    *,
    thresholds: RecommendationThresholds,
    comparison_policy_id: str,
) -> None:
    data = summary.copy()
    annotated = set(focus_policy_ids(summary, comparison_policy_id=comparison_policy_id)[:2])
    data["focus"] = data["label"].where(data["policy_id"].isin(annotated), "")
    data["Top-k status"] = (
        data["min_effective_topk"]
        .ge(thresholds.min_effective_topk)
        .map({True: "Full top-k", False: "Incomplete top-k"})
    )
    data["Shared by all target views"] = data["all_target_views_overlap"]
    fig = plt.figure(figsize=(9.4, 5.2))
    ax = sns.scatterplot(
        data=data,
        x="min_target_view_median_logic",
        y="mean_topk_effect",
        hue="Shared by all target views",
        style="Top-k status",
        s=78,
        palette="viridis_r",
        edgecolor="#333333",
        linewidth=0.4,
    )
    ax.axvline(thresholds.min_target_view_median_logic, color="#b94a48", linestyle="--", linewidth=1.2)
    ax.text(
        thresholds.min_target_view_median_logic + 0.003,
        float(data["mean_topk_effect"].max()) * 0.95,
        "logic guardrail",
        color="#8b2e2c",
        fontsize=8,
    )
    for _, row in data[data["focus"] != ""].iterrows():
        ax.text(
            float(row["min_target_view_median_logic"]) + 0.003,
            float(row["mean_topk_effect"]) + 0.003,
            str(row["focus"]),
            fontsize=7,
        )
    ax.set_xlabel("Weakest-target-view median top-k logic fidelity")
    ax.set_ylabel("Mean selected SFXI scaled effect")
    ax.set_title("Policy decision frontier")
    ax.set_box_aspect(1)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    plt.tight_layout()
    save_metastudy_figure(fig, path)


def write_score_component_dominance(
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    path: Path,
    *,
    comparison_policy_id: str,
) -> None:
    focus = set(focus_policy_ids(summary, comparison_policy_id=comparison_policy_id)[:2])
    data = pairwise[
        (pairwise["policy_id"].isin(focus))
        & (pairwise["metric"] == "within_selection_view")
        & (pairwise["selection_view_b"].isin(["logic_fidelity", "effect_scaled"]))
    ].copy()
    data = data.rename(columns={"selection_view_a": "selection_view_id", "selection_view_b": "component"})
    data["component"] = data["component"].replace(
        {"logic_fidelity": "Score vs logic fidelity", "effect_scaled": "Score vs scaled effect"}
    )
    data["Target view"] = data["selection_view_id"].map(target_view_label)
    data["Policy"] = data["policy_id"].map(policy_label)
    data["Score component"] = data["component"]
    grid = sns.catplot(
        data=data,
        x="Target view",
        y="pearson",
        hue="Score component",
        col="Policy",
        kind="bar",
        height=3.5,
        aspect=1.0,
        palette=["#4f8a5b", "#b94a48"],
    )
    grid.set_axis_labels("Target view", "Pearson correlation")
    grid.set_titles("{col_name}")
    for ax in grid.axes.flat:
        ax.axhline(0.0, color="#444444", linewidth=0.8)
        ax.set_ylim(-0.7, 1.05)
        ax.set_box_aspect(1)
        ax.tick_params(axis="x", rotation=25)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    grid.fig.suptitle("Score component dominance", y=1.08)
    grid.tight_layout()
    save_metastudy_figure(grid.fig, path)
