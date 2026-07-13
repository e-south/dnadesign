"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/screen_plots.py

Appendix screen plots for the response metric metastudy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .plot_helpers import focus_policy_ids
from .plot_style import save_metastudy_figure
from .plot_vocabulary import compact_policy_label


def write_logic_effect_tradeoff_overlap(summary: pd.DataFrame, path: Path) -> None:
    sweep = _tradeoff_screen(summary)
    fig = plt.figure(figsize=(7.8, 4.5))
    sns.lineplot(data=sweep, x="logic_tradeoff_weight", y="unique_topk", marker="o", color="#2A6F97")
    plt.title("Top-k uniqueness across the logic-effect tradeoff")
    axis = plt.gca()
    axis.set_xlabel("Normalized logic weight")
    axis.set_ylabel("Unique candidates across target-view selections")
    axis.set_box_aspect(1)
    plt.tight_layout()
    save_metastudy_figure(fig, path)


def write_logic_effect_tradeoff_fidelity(summary: pd.DataFrame, path: Path) -> None:
    sweep = _tradeoff_screen(summary)
    fig = plt.figure(figsize=(7.8, 4.5))
    sns.lineplot(
        data=sweep,
        x="logic_tradeoff_weight",
        y="min_target_view_median_logic",
        marker="o",
        color="#B23A48",
    )
    plt.title("Selected target-shape fidelity across the logic-effect tradeoff")
    axis = plt.gca()
    axis.set_xlabel("Normalized logic weight")
    axis.set_ylabel("Weakest-target-view median logic fidelity")
    axis.set_box_aspect(1)
    plt.tight_layout()
    save_metastudy_figure(fig, path)


def _tradeoff_screen(summary: pd.DataFrame) -> pd.DataFrame:
    screen = summary[(summary["kind"] == "multiplicative") & (summary["tier"].isin(["canonical", "sweep"]))].copy()
    total = screen["beta"] + screen["gamma"]
    if screen.empty or (total <= 0.0).any():
        raise ValueError("multiplicative tradeoff screen requires positive beta + gamma.")
    screen["logic_tradeoff_weight"] = screen["beta"] / total
    return screen.sort_values("logic_tradeoff_weight", kind="mergesort")


def write_policy_overlap_summary(
    summary: pd.DataFrame,
    path: Path,
    *,
    comparison_policy_id: str,
) -> None:
    focus = focus_policy_ids(summary, comparison_policy_id=comparison_policy_id)
    data = summary[summary["policy_id"].isin(focus)].copy()
    data["label_display"] = data["policy_id"].map(compact_policy_label)
    fig = plt.figure(figsize=(9.2, 4.8))
    ax = sns.barplot(data=data, x="label_display", y="unique_topk", color="#4c78a8")
    for idx, row in data.reset_index(drop=True).iterrows():
        ax.text(
            idx,
            float(row["unique_topk"]) + 0.25,
            f"Top-k: {int(row['min_effective_topk'])}\n"
            f"Shared by all: {int(row['all_target_views_overlap'])}\n"
            f"Pairwise reuse: {int(row['pairwise_overlap_total'])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, 19)
    ax.set_xlabel("")
    ax.set_ylabel("Unique sequences across target-view top-k selections")
    ax.set_title("Focus policy overlap summary")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    save_metastudy_figure(fig, path)


def write_topk_overlap_curve(
    summary: pd.DataFrame,
    overlap_by_k: pd.DataFrame,
    path: Path,
    *,
    comparison_policy_id: str,
) -> None:
    focus = focus_policy_ids(summary, comparison_policy_id=comparison_policy_id)
    data = overlap_by_k[
        (overlap_by_k["policy_id"].isin(focus)) & (overlap_by_k["overlap_type"] == "all_target_views")
    ].copy()
    data["Policy"] = data["policy_id"].map(compact_policy_label)
    fig = plt.figure(figsize=(7.8, 4.5))
    ax = sns.lineplot(data=data, x="k", y="observed_overlap", hue="Policy", marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("Candidates retained per target view (K)")
    ax.set_ylabel("Candidates shared by all target views")
    ax.set_title("All-target-view overlap across K")
    ax.set_box_aspect(1)
    ax.legend(fontsize=8)
    plt.tight_layout()
    save_metastudy_figure(fig, path)
