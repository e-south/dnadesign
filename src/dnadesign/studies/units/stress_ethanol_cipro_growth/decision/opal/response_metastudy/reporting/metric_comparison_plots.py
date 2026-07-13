"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/metric_comparison_plots.py

Didactic comparison plots for SFXI and RMF components.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D

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


def write_metric_compensation_comparison(frame: pd.DataFrame, path: Path) -> None:
    _require(frame)
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(12.8, 8.6),
        sharex="row",
        sharey="row",
        constrained_layout=True,
    )
    sfxi_min = float(frame["sfxi"].min())
    sfxi_max = float(frame["sfxi"].max())
    if np.isclose(sfxi_min, sfxi_max):
        sfxi_max = sfxi_min + 1.0e-9
    sfxi_norm = Normalize(vmin=sfxi_min, vmax=sfxi_max)
    off_extent = max(float(frame["off_suppression"].abs().max()), 1.0e-9)
    off_norm = TwoSlopeNorm(vmin=-off_extent, vcenter=0.0, vmax=off_extent)
    sfxi_scatter = None
    response_scatter = None
    for column, selection_view_id in enumerate(_TARGET_VIEW_IDS):
        rows = frame.loc[frame["selection_view_id"].astype(str).eq(selection_view_id)].copy()
        if rows.empty:
            raise ValueError(f"metric comparison lacks target view {selection_view_id!r}.")
        top = axes[0, column]
        sfxi_scatter = top.scatter(
            rows["logic_fidelity"],
            rows["effect_scaled"],
            c=rows["sfxi"],
            cmap="viridis",
            norm=sfxi_norm,
            edgecolors="#ffffff",
            linewidths=0.5,
            alpha=0.65,
            s=34,
            zorder=3,
        )
        top.set_title(_TARGET_VIEW_LABELS[selection_view_id])
        top.set_xlabel("SFXI logic fidelity\nHigher is better" if column == 1 else "")
        top.set_ylabel("SFXI scaled effect\nHigher is better" if column == 0 else "")
        top.set_box_aspect(1)

        bottom = axes[1, column]
        response_scatter = bottom.scatter(
            rows["response_separation"],
            rows["on_magnitude_floor"],
            c=rows["off_suppression"],
            cmap="RdBu",
            norm=off_norm,
            edgecolors=np.where(rows["passes_all_zero_constraints"].astype(bool), "#111827", "#ffffff"),
            linewidths=np.where(rows["passes_all_zero_constraints"].astype(bool), 1.8, 0.5),
            alpha=0.65,
            s=34,
            zorder=3,
        )
        bottom.axvline(0.0, color="#6b7280", linestyle="--", linewidth=0.9)
        bottom.axhline(0.0, color="#6b7280", linestyle="--", linewidth=0.9)
        bottom.set_xlabel(
            "RMF response margin\nweakest ON - strongest OFF log2(YFP / CFP)\nHigher is better" if column == 1 else ""
        )
        bottom.set_ylabel(
            "RMF ON-fluorescence margin\nweakest ON relative to pDual-10\nHigher is better" if column == 0 else ""
        )
        bottom.set_box_aspect(1)
        _annotate_examples(top, rows, x="logic_fidelity", y="effect_scaled")
        _annotate_examples(bottom, rows, x="response_separation", y="on_magnitude_floor")
    if sfxi_scatter is None or response_scatter is None:
        raise RuntimeError("metric comparison did not render any target views.")
    figure.colorbar(
        sfxi_scatter,
        ax=axes[0, :],
        shrink=0.78,
        label="Canonical SFXI score\nHigher is better",
    )
    figure.colorbar(
        response_scatter,
        ax=axes[1, :],
        shrink=0.78,
        label="RMF OFF-control margin\nnegative of strongest OFF fluorescence vs pDual-10\nHigher is better",
    )
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markerfacecolor="none",
                markeredgecolor="#111827",
                markeredgewidth=1.8,
                label="Passes all provisional zero boundaries",
            )
        ],
        loc="outside lower center",
        frameon=False,
    )
    save_metastudy_figure(figure, path)


def write_measured_response_examples(frame: pd.DataFrame, path: Path) -> None:
    _require(frame)
    examples = frame.loc[frame["is_response_example"].astype(bool)].copy()
    if examples.empty:
        raise ValueError("measured response plot has no configured examples.")
    labels = list(dict.fromkeys(examples["example_label"].astype(str)))
    figure = plt.figure(figsize=(12.6, 9.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(1.08, 1.0))
    raw_axes: list[plt.Axes] = []
    component_axes: list[plt.Axes] = []
    width = 0.24
    colors = ("#2563eb", "#0f766e", "#be123c")
    raw_columns = [f"{prefix}{state}" for prefix in ("r", "b") for state in _STATE_IDS]
    raw_limit = max(float(np.quantile(np.abs(examples.loc[:, raw_columns].to_numpy(dtype=float)), 0.98)), 1.0)
    raw_image = None
    for column_index, selection_view_id in enumerate(_TARGET_VIEW_IDS):
        rows = (
            examples.loc[examples["selection_view_id"].astype(str).eq(selection_view_id)]
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
        raise RuntimeError("measured response plot did not render any target view.")
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


def _annotate_examples(axis: plt.Axes, rows: pd.DataFrame, *, x: str, y: str) -> None:
    for row in rows.loc[rows["is_response_example"].astype(bool)].itertuples(index=False):
        axis.annotate(
            _short_example(row.example_label),
            (float(getattr(row, x)), float(getattr(row, y))),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )


def _short_example(value: object) -> str:
    return str(value).split()[0]


def _require(frame: pd.DataFrame) -> None:
    required = {
        "selection_view_id",
        "logic_fidelity",
        "effect_scaled",
        "sfxi",
        "response_separation",
        "on_magnitude_floor",
        "off_magnitude_ceiling",
        "off_suppression",
        "passes_all_zero_constraints",
        "example_label",
        "is_response_example",
        *(f"{prefix}{state}" for prefix in ("r", "b") for state in _STATE_IDS),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"metric comparison plot missing columns: {missing}")


__all__ = ["write_measured_response_examples", "write_metric_compensation_comparison"]
