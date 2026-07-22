"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/response_model_plots.py

Plots for grouped response-label modeling and retrospective enrichment.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .matrix_annotations import label_representation_axis
from .plot_helpers import contrast_text_color, ordered_pivot, require_columns
from .plot_style import save_metastudy_figure
from .plot_vocabulary import REPRESENTATION_ORDER as _REPRESENTATION_ORDER
from .plot_vocabulary import TARGET_VIEW_LABELS as _TARGET_VIEW_LABELS
from .plot_vocabulary import TARGET_VIEW_ORDER as _TARGET_VIEW_ORDER
from .plot_vocabulary import target_view_label

_MODEL_ORDER = (
    "mean_baseline",
    "campaign_random_forest",
    "robust_target_random_forest",
    "pca4_ridge10",
    "pca8_ridge10",
    "pca12_ridge10",
    "pls2",
    "pls4",
    "pls6",
)
_MODEL_LABELS = {
    "mean_baseline": "Mean baseline",
    "campaign_random_forest": "Configured campaign RF",
    "robust_target_random_forest": "Robust target RF",
    "pca4_ridge10": "PCA-4 + ridge",
    "pca8_ridge10": "PCA-8 + ridge",
    "pca12_ridge10": "PCA-12 + ridge",
    "pls2": "PLS-2",
    "pls4": "PLS-4",
    "pls6": "PLS-6",
}


def write_greedy_support_evidence(frame: pd.DataFrame, path: Path) -> None:
    require_columns(
        frame,
        {
            "selection_view_id",
            "fraction_beating_group_median",
            "fraction_ci_low",
            "fraction_ci_high",
            "groups_beating_median",
            "held_out_group_count",
        },
        context="greedy-support plot",
    )
    work = frame.set_index("selection_view_id").reindex(
        [value for value in _TARGET_VIEW_ORDER if value in set(frame["selection_view_id"])]
    )
    values = work["fraction_beating_group_median"].to_numpy(dtype=float)
    lower = values - work["fraction_ci_low"].to_numpy(dtype=float)
    upper = work["fraction_ci_high"].to_numpy(dtype=float) - values
    x = np.arange(len(work))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.errorbar(x, values, yerr=np.vstack((lower, upper)), fmt="o", color="#2f5597", capsize=5, linewidth=2)
    ax.axhline(0.5, color="#6b7280", linestyle="--", linewidth=1)
    ax.set_xticks(x, [_TARGET_VIEW_LABELS[str(value)] for value in work.index])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Fraction of held-out Reader experiments")
    ax.set_box_aspect(1.0)
    for index, row in enumerate(work.itertuples()):
        anchor_y = min(1.0, float(row.fraction_ci_high))
        near_top = anchor_y >= 0.90
        horizontal_offset = 8 if index == 0 else -8 if index == len(work) - 1 else 0
        ax.annotate(
            f"{int(row.groups_beating_median)}/{int(row.held_out_group_count)} groups",
            (index, anchor_y),
            xytext=(horizontal_offset, -8 if near_top else 8),
            textcoords="offset points",
            ha="left" if index == 0 else "right" if index == len(work) - 1 else "center",
            va="top" if near_top else "bottom",
            fontsize=11,
        )
    fig.tight_layout()
    save_metastudy_figure(fig, path)


def write_label_model_screen(frame: pd.DataFrame, path: Path) -> None:
    required = {
        "representation_id",
        "model_id",
        "promotion_eligible",
        "weakest_target_view_response_separation_spearman",
        "weakest_target_view_feasibility_spearman",
        "weakest_required_ordering_spearman",
    }
    require_columns(frame, required, context="label model screen plot")
    pivot = frame.pivot(
        index="model_id",
        columns="representation_id",
        values="weakest_required_ordering_spearman",
    )
    pivot = ordered_pivot(pivot, rows=_MODEL_ORDER, columns=_REPRESENTATION_ORDER)
    values = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(14.4, 9.6))
    image = ax.imshow(values, cmap="coolwarm", vmin=-0.4, vmax=0.4, aspect="equal")
    label_representation_axis(ax, pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), [_MODEL_LABELS[str(value)] for value in pivot.index])
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            label = f"{values[row, column]:.2f}" if np.isfinite(values[row, column]) else "--"
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color=contrast_text_color(image, values[row, column]),
            )
    ax.set_title("Grouped model screen across response representations", pad=34)
    ax.set_xlabel("Response representation")
    ax.set_ylabel("Fixed model screen")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.03)
    colorbar.set_label("Median within-experiment Spearman correlation")
    fig.tight_layout()
    save_metastudy_figure(fig, path)


def write_retrospective_enrichment(
    summary: pd.DataFrame,
    model_screen: pd.DataFrame,
    path: Path,
) -> None:
    required = {
        "representation_id",
        "model_id",
        "selection_view_id",
        "median_selected_true_percentile",
        "promotion_eligible",
    }
    require_columns(summary, required, context="retrospective enrichment plot")
    eligible_models = model_screen.loc[
        model_screen["promotion_eligible"].astype(bool) & model_screen["all_target_view_metrics_finite"].astype(bool)
    ].copy()
    require_columns(
        eligible_models,
        {"weakest_required_ordering_spearman"},
        context="retrospective enrichment model screen",
    )
    best = (
        eligible_models.sort_values(
            ["representation_id", "weakest_required_ordering_spearman"],
            ascending=[True, False],
            kind="mergesort",
        )
        .groupby("representation_id", sort=True, as_index=False)
        .head(1)
        .loc[:, ["representation_id", "model_id"]]
    )
    work = summary.merge(best, on=["representation_id", "model_id"], how="inner", validate="many_to_one")
    pivot = work.pivot(
        index="selection_view_id",
        columns="representation_id",
        values="median_selected_true_percentile",
    )
    pivot = ordered_pivot(pivot, rows=_TARGET_VIEW_ORDER, columns=_REPRESENTATION_ORDER)
    values = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(13.2, 5.8))
    image = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
    label_representation_axis(ax, pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), [target_view_label(value) for value in pivot.index])
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                f"{values[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color=contrast_text_color(image, values[row, column]),
            )
    ax.set_title("Retrospective held-out selection percentile for each label representation", pad=34)
    ax.set_xlabel("Best fixed challenger within representation")
    ax.set_ylabel("Selection target view")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.03)
    colorbar.set_label("Median true percentile within held-out experiment")
    fig.tight_layout()
    save_metastudy_figure(fig, path)


__all__ = [
    "write_greedy_support_evidence",
    "write_label_model_screen",
    "write_retrospective_enrichment",
]
