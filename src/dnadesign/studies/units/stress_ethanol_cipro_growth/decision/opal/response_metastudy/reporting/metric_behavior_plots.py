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

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from ..core.contracts import SfxiEvidenceFrame
from ..core.policies import CANONICAL_SFXI_POLICY_ID
from .plot_helpers import focus_policy_ids
from .plot_style import save_metastudy_figure
from .plot_vocabulary import (
    panel_role_label,
    policy_label,
    prediction_component_label,
    selection_layer_label,
    target_view_label,
)


def write_sfxi_score_contours(
    summary: pd.DataFrame,
    path: Path,
    *,
    score_surface_policy_id: str,
) -> None:
    focus = _score_surface_policy_rows(summary, score_surface_policy_id=score_surface_policy_id)
    logic = np.linspace(0.0, 1.0, 81)
    effect = np.linspace(0.0, 1.0, 81)
    logic_grid, effect_grid = np.meshgrid(logic, effect)
    fig, axes = plt.subplots(1, len(focus), figsize=(5.2 * len(focus), 4.4), squeeze=False)
    for ax, (_, policy) in zip(axes[0], focus.iterrows(), strict=True):
        score = np.power(logic_grid, float(policy["beta"])) * np.power(effect_grid, float(policy["gamma"]))
        contour = ax.contourf(logic_grid, effect_grid, score, levels=12, cmap="viridis")
        ax.contour(logic_grid, effect_grid, score, levels=6, colors="white", linewidths=0.45, alpha=0.7)
        ax.set_title(policy_label(policy["policy_id"]))
        ax.set_xlabel("SFXI logic fidelity")
        ax.set_ylabel("SFXI scaled effect")
        ax.set_box_aspect(1)
        fig.colorbar(contour, ax=ax, label="score")
    fig.suptitle("SFXI score contours", y=1.02)
    fig.tight_layout()
    save_metastudy_figure(fig, path)


def write_target_view_pareto_fronts(
    summary: pd.DataFrame,
    scored: dict[str, dict[str, pd.DataFrame]],
    sfxi_evidence: tuple[SfxiEvidenceFrame, ...],
    path: Path,
    *,
    comparison_policy_id: str,
) -> None:
    focus = focus_policy_ids(summary, comparison_policy_id=comparison_policy_id)[:2]
    canonical_policy = CANONICAL_SFXI_POLICY_ID
    rows: list[pd.DataFrame] = []
    for evidence in sfxi_evidence:
        frame = scored[canonical_policy][evidence.target_view.id]
        sample = frame.sample(n=min(2500, len(frame)), random_state=17).copy()
        sample["layer"] = "candidate sample"
        sample["plot_policy"] = canonical_policy
        rows.append(sample)
        for policy_id in focus:
            selected = scored[policy_id][evidence.target_view.id].head(6).copy()
            selected["layer"] = "selected top-6"
            selected["plot_policy"] = policy_id
            rows.append(selected)
    data = pd.concat(rows, ignore_index=True)
    data["Target view"] = data["selection_view_id"].map(target_view_label)
    data["Policy"] = data["plot_policy"].map(policy_label)
    data["Selection layer"] = data["layer"].map(selection_layer_label)
    grid = sns.relplot(
        data=data,
        x="logic_fidelity",
        y="effect_scaled",
        col="Target view",
        hue="Policy",
        style="Selection layer",
        kind="scatter",
        height=3.4,
        aspect=1.0,
        alpha=0.72,
        s=28,
    )
    grid.set_axis_labels("SFXI logic fidelity", "SFXI scaled effect")
    grid.set_titles("{col_name}")
    for axis in grid.axes.flat:
        axis.set_box_aspect(1)
    sns.move_legend(grid, "center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    grid.fig.suptitle("Target-view logic/effect candidate surfaces", y=1.05)
    grid.tight_layout()
    save_metastudy_figure(grid.fig, path)


def write_denominator_sensitivity(denominator_sensitivity: pd.DataFrame, path: Path) -> None:
    data = denominator_sensitivity.melt(
        id_vars=["policy_id", "selection_view_id", "denominator_factor"],
        value_vars=["median_logic_fidelity", "median_effect_scaled"],
        var_name="metric",
        value_name="value",
    )
    data["Target view"] = data["selection_view_id"].map(target_view_label)
    data["Policy"] = data["policy_id"].map(policy_label)
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
    grid.set_axis_labels("SFXI denominator scale factor", "Top-k median")
    grid.set(ylim=(0.0, 1.0))
    grid.set_titles("{col_name}")
    for axis in grid.axes.flat:
        axis.set_box_aspect(1)
    sns.move_legend(grid, "center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    grid.fig.suptitle("Denominator sensitivity", y=1.05)
    grid.tight_layout()
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
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    save_metastudy_figure(fig, path)


def _score_surface_policy_rows(summary: pd.DataFrame, *, score_surface_policy_id: str) -> pd.DataFrame:
    focus = [CANONICAL_SFXI_POLICY_ID, score_surface_policy_id]
    missing_ids = sorted(set(focus) - set(summary["policy_id"].astype(str)))
    if missing_ids:
        raise ValueError(f"Score contour policies are absent from summary: {missing_ids}")
    rows = summary[summary["policy_id"].isin(focus)].drop_duplicates("policy_id")
    rows = rows.set_index("policy_id").loc[focus].reset_index()
    supported = rows[rows["kind"].isin({"multiplicative", "off_state_logic_penalty"})]
    if len(supported) != len(rows):
        unsupported = sorted(set(rows["policy_id"]) - set(supported["policy_id"]))
        raise ValueError(f"Score contour plot only supports scalar policies; unsupported: {unsupported}")
    return supported
