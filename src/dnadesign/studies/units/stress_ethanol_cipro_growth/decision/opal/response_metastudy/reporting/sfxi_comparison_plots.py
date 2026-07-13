"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/sfxi_comparison_plots.py

Plots comparing canonical SFXI across Reader-owned assay reductions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_style import save_metastudy_figure
from .plot_vocabulary import SFXI_COMPARISON_ORDER, representation_label, target_view_label


def write_sfxi_comparison_stability(frame: pd.DataFrame, path: Path) -> None:
    required = {"assay_summary_id", "selection_view_id", "score_spearman_to_snapshot"}
    _require_columns(frame, required, context="assay summary stability plot")
    pivot = frame.pivot(
        index="selection_view_id",
        columns="assay_summary_id",
        values="score_spearman_to_snapshot",
    )
    pivot = _ordered_summary_columns(pivot)
    values = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    finite = values[np.isfinite(values)]
    lower_bound = max(-1.0, min(0.9, float(np.floor(finite.min() * 10.0) / 10.0)))

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    image = ax.imshow(masked, cmap="viridis", vmin=lower_bound, vmax=1.0, aspect="equal")
    ax.set_xticks(
        np.arange(len(pivot.columns)),
        [representation_label(value) for value in pivot.columns],
        ha="center",
        fontsize=7,
    )
    ax.set_yticks(np.arange(len(pivot.index)), [target_view_label(value) for value in pivot.index])
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            label = "NA" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(column, row, label, ha="center", va="center", fontsize=8, color="#111827")
    ax.set_title("Observed SFXI rank agreement with the 12 h snapshot")
    ax.set_xlabel("Assay summary")
    ax.set_ylabel("Target view")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.03)
    colorbar.set_label("Spearman correlation")
    fig.tight_layout()
    _write_figure(fig, path)


def write_sfxi_comparison_target_coverage(
    frame: pd.DataFrame,
    path: Path,
    *,
    logic_threshold: float,
) -> None:
    required = {"assay_summary_id", "selection_view_id", "logic_support_count", "median_logic_fidelity"}
    _require_columns(frame, required, context="assay target-coverage plot")
    summaries = _ordered_summary_ids(frame["assay_summary_id"].astype(str).unique().tolist())
    target_view_ids = sorted(frame["selection_view_id"].astype(str).unique())
    x_index = {value: index for index, value in enumerate(summaries)}
    y_index = {value: index for index, value in enumerate(target_view_ids)}

    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    counts = frame["logic_support_count"].to_numpy(dtype=float)
    colors = frame["median_logic_fidelity"].to_numpy(dtype=float)
    scatter = ax.scatter(
        [x_index[str(value)] for value in frame["assay_summary_id"]],
        [y_index[str(value)] for value in frame["selection_view_id"]],
        s=55.0 + 8.0 * counts,
        c=colors,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        edgecolors="#111827",
        linewidths=0.5,
    )
    for _, row in frame.iterrows():
        ax.text(
            x_index[str(row["assay_summary_id"])],
            y_index[str(row["selection_view_id"])],
            str(int(row["logic_support_count"])),
            ha="center",
            va="center",
            fontsize=8,
            color="white" if float(row["median_logic_fidelity"]) < 0.55 else "#111827",
        )
    ax.set_xticks(
        np.arange(len(summaries)),
        [representation_label(value) for value in summaries],
        ha="center",
        fontsize=7,
    )
    ax.set_yticks(
        np.arange(len(target_view_ids)),
        [target_view_label(value) for value in target_view_ids],
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Assay summary")
    ax.set_ylabel("Target view")
    ax.set_title(f"Observed designs above the provisional {logic_threshold:.2f} logic-fidelity review line")
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.03)
    colorbar.set_label("Median observed logic fidelity")
    fig.tight_layout()
    _write_figure(fig, path)


def _ordered_summary_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, _ordered_summary_ids([str(value) for value in frame.columns])]


def _ordered_summary_ids(values: list[str]) -> list[str]:
    unknown = sorted(set(values) - set(SFXI_COMPARISON_ORDER))
    if unknown:
        raise ValueError(f"SFXI comparison contains unregistered assay summaries: {unknown}")
    return [value for value in SFXI_COMPARISON_ORDER if value in values]


def _require_columns(frame: pd.DataFrame, required: set[str], *, context: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{context} missing columns: {missing}")


def _write_figure(fig: plt.Figure, path: Path) -> None:
    save_metastudy_figure(fig, path)


__all__ = ["write_sfxi_comparison_stability", "write_sfxi_comparison_target_coverage"]
