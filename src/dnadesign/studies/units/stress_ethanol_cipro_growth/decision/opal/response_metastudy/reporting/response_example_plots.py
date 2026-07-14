"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/response_example_plots.py

Measured response-window examples under declared stress target masks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_style import save_metastudy_figure

_TARGET_VIEW_IDS = ("ethanol", "ciprofloxacin", "and")
_TARGET_VIEW_LABELS = {"ethanol": "Ethanol", "ciprofloxacin": "Ciprofloxacin", "and": "AND"}
_TARGET_VIEW_ON_LABELS = {
    "ethanol": "ON: ethanol; ethanol + ciprofloxacin",
    "ciprofloxacin": "ON: ciprofloxacin; ethanol + ciprofloxacin",
    "and": "ON: ethanol + ciprofloxacin only",
}
_TARGET_VIEW_MASKS = {
    "ethanol": (0, 1, 0, 1),
    "ciprofloxacin": (0, 0, 1, 1),
    "and": (0, 0, 0, 1),
}
_STATE_LABELS = ("No stress", "Ethanol", "Ciprofloxacin", "Both stresses")
_STATE_IDS = ("00", "10", "01", "11")


def write_measured_response_examples(frame: pd.DataFrame, path: Path) -> None:
    """Plot fixed Reader response values and their target-mask components."""

    _require(frame)
    labels = list(dict.fromkeys(frame["example_label"].astype(str)))
    if not labels:
        raise ValueError("measured response plot has no configured examples.")
    figure = plt.figure(figsize=(12.6, 9.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(1.08, 1.0))
    raw_axes: list[plt.Axes] = []
    component_axes: list[plt.Axes] = []
    width = 0.24
    colors = ("#2563eb", "#0f766e", "#be123c")
    raw_columns = [f"{prefix}{state}" for prefix in ("r", "b") for state in _STATE_IDS]
    raw_limit = max(float(np.quantile(np.abs(frame.loc[:, raw_columns].to_numpy(dtype=float)), 0.98)), 1.0)
    raw_image = None
    for column_index, selection_view_id in enumerate(_TARGET_VIEW_IDS):
        rows = (
            frame.loc[frame["selection_view_id"].astype(str).eq(selection_view_id)]
            .set_index("example_label")
            .reindex(labels)
        )
        required_values = [*raw_columns, "response_separation", "on_magnitude_floor", "off_suppression"]
        if rows[required_values].isna().any().any():
            raise ValueError(f"measured response rows are incomplete for target view {selection_view_id!r}.")

        raw_axis = figure.add_subplot(grid[0, column_index])
        raw_axes.append(raw_axis)
        raw_matrix = np.vstack(
            [
                rows.loc[label, [f"{prefix}{state}" for state in _STATE_IDS]].to_numpy(dtype=float)
                for label in labels
                for prefix in ("r", "b")
            ]
        )
        raw_image = raw_axis.imshow(raw_matrix, cmap="coolwarm", vmin=-raw_limit, vmax=raw_limit, aspect="equal")
        raw_axis.set_xticks(
            np.arange(4), [f"{state}\n{label}" for state, label in zip(_STATE_IDS, _STATE_LABELS, strict=True)]
        )
        raw_axis.tick_params(axis="x", labelrotation=42, labelsize=7)
        raw_axis.set_yticks(
            np.arange(len(labels) * 2),
            [
                f"{_short_example(label)}\n{measurement}"
                for label in labels
                for measurement in ("log2(YFP / CFP)", "relative fluorescence")
            ],
            fontsize=7,
        )
        mask_axis = raw_axis.secondary_xaxis("top")
        mask = _TARGET_VIEW_MASKS[selection_view_id]
        mask_axis.set_xticks(np.arange(4), ["ON" if value else "OFF" for value in mask])
        mask_axis.tick_params(axis="x", length=0, pad=4, labelsize=8)
        for tick, value in zip(mask_axis.get_xticklabels(), mask, strict=True):
            tick.set_color("#047857" if value else "#6b7280")
            tick.set_fontweight("semibold")
        mask_axis.spines["top"].set_visible(False)
        raw_axis.set_title(
            f"{_TARGET_VIEW_LABELS[selection_view_id]} mask {list(mask)}\n{_TARGET_VIEW_ON_LABELS[selection_view_id]}",
            fontsize=9,
            pad=34,
        )
        for row_index, column in np.ndindex(raw_matrix.shape):
            raw_axis.text(
                column,
                row_index,
                f"{raw_matrix[row_index, column]:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(raw_matrix[row_index, column]) > raw_limit * 0.55 else "#111827",
            )

        axis = figure.add_subplot(grid[1, column_index], sharey=component_axes[0] if component_axes else None)
        component_axes.append(axis)
        x = np.arange(len(labels))
        for offset, (column, label, color) in enumerate(
            zip(
                ("response_separation", "on_magnitude_floor", "off_suppression"),
                (
                    "Response: min ON - max OFF",
                    "ON fluorescence: min relative to pDual-10",
                    "OFF fluorescence: -max relative to pDual-10",
                ),
                colors,
                strict=True,
            )
        ):
            axis.bar(x + (offset - 1) * width, rows[column], width, label=label, color=color, zorder=3)
        axis.axhline(0.0, color="#111827", linewidth=0.9)
        axis.set_xticks(x, [_short_example(value) for value in labels])
        axis.set_box_aspect(1)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#e5e7eb", linewidth=0.7, zorder=0)
    if raw_image is None:
        raise RuntimeError("measured response plot did not render any target views.")
    component_axes[0].set_ylabel(
        "Unscaled RMF requirement (log2 units)\nHigher is better; 0 is the provisional boundary"
    )
    handles, legend_labels = component_axes[-1].get_legend_handles_labels()
    figure.colorbar(
        raw_image,
        ax=raw_axes,
        location="right",
        shrink=0.78,
        label="Measured window summary (log2 units)",
    )
    figure.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="outside lower center",
        ncols=3,
        title="Reader values are fixed; the target-view mask changes the RMF components below",
    )
    save_metastudy_figure(figure, path)


def _short_example(value: object) -> str:
    return str(value).split()[0]


def _require(frame: pd.DataFrame) -> None:
    required = {
        "selection_view_id",
        "response_separation",
        "on_magnitude_floor",
        "off_magnitude_ceiling",
        "off_suppression",
        "passes_all_zero_constraints",
        "example_label",
        *(f"{prefix}{state}" for prefix in ("r", "b") for state in _STATE_IDS),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"measured response plot missing columns: {missing}")


__all__ = ["write_measured_response_examples"]
