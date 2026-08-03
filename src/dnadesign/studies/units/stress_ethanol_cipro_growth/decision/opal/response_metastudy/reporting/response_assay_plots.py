"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/response_assay_plots.py

Plots for Reader event evidence, response constraints, and repeated measurements.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .matrix_annotations import add_column_group_brackets, label_representation_axis
from .plot_helpers import contrast_text_color, ordered_pivot, require_columns
from .plot_style import save_metastudy_figure
from .plot_vocabulary import (
    REDUCTION_ORDER as _REDUCTION_ORDER,
)
from .plot_vocabulary import (
    STATE_TICK_LABELS as _STATE_TICK_LABELS,
)
from .plot_vocabulary import (
    TARGET_VIEW_LABELS as _TARGET_VIEW_LABELS,
)
from .plot_vocabulary import (
    TARGET_VIEW_ORDER as _TARGET_VIEW_ORDER,
)
from .plot_vocabulary import (
    reader_experiment_label,
    representation_label,
)

_TARGET_VIEW_MASK_LABELS = {
    "ethanol": "Ethanol target\nON: ethanol + both",
    "ciprofloxacin": "Ciprofloxacin target\nON: ciprofloxacin + both",
    "and": "AND target\nON: both only",
    "or": "OR screen\nON: either stress or both",
}
_STATE_LABELS = {
    "00": "No stress",
    "10": "Ethanol",
    "01": "Ciprofloxacin",
    "11": "Ethanol + ciprofloxacin",
}


def write_reader_event_intervals(frame: pd.DataFrame, path: Path) -> None:
    require_columns(
        frame,
        {
            "experiment_id",
            "event_time_uncertainty_h",
            "post_event_coverage_h",
        },
        context="Reader event interval plot",
    )
    work = frame.sort_values("event_time_uncertainty_h", kind="mergesort")
    y = np.arange(len(work))
    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    bars = ax.barh(y, 2.0 * work["event_time_uncertainty_h"], color="#4c78a8")
    ax.set_yticks(y, [reader_experiment_label(value) for value in work["experiment_id"]])
    ax.set_xlabel("Unresolved stress-addition interval (h)")
    ax.set_ylabel("Reader experiment")
    ax.set_title("Declared intervention interval by Reader experiment")
    for bar, coverage in zip(
        bars,
        work["post_event_coverage_h"],
        strict=True,
    ):
        ax.text(
            bar.get_width() + 0.015,
            bar.get_y() + bar.get_height() / 2.0,
            f"{float(coverage):.1f} h post-event coverage",
            va="center",
            fontsize=10,
        )
    ax.set_xlim(0.0, max(0.8, float((2.0 * work["event_time_uncertainty_h"]).max()) + 0.35))
    fig.tight_layout()
    save_metastudy_figure(fig, path)


def write_response_separation_stability(
    frame: pd.DataFrame,
    path: Path,
    *,
    primary_reduction_id: str,
) -> None:
    correlation_columns = [
        "response_separation__spearman_to_primary",
        "on_magnitude_floor__spearman_to_primary",
        "off_magnitude_ceiling__spearman_to_primary",
    ]
    require_columns(
        frame,
        {"reduction_id", "selection_view_id", *correlation_columns},
        context="margin stability plot",
    )
    work = frame.copy()
    work["weakest_component_spearman"] = work[correlation_columns].min(axis=1)
    pivot = work.pivot(index="selection_view_id", columns="reduction_id", values="weakest_component_spearman")
    pivot = ordered_pivot(pivot, rows=_TARGET_VIEW_ORDER, columns=_REDUCTION_ORDER)
    values = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(13.2, 7.0))
    image = ax.imshow(
        values,
        cmap="viridis",
        vmin=max(-1.0, float(np.nanmin(values))),
        vmax=1.0,
        aspect="equal",
    )
    label_representation_axis(ax, pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), [_TARGET_VIEW_LABELS[str(value)] for value in pivot.index])
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
    primary_label = representation_label(primary_reduction_id).replace("\n", " ")
    ax.set_title(f"Weakest component rank agreement with the {primary_label}", pad=34)
    ax.set_xlabel("Response reduction")
    ax.set_ylabel("Target mask")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.03)
    colorbar.set_label("Weakest component Spearman correlation")
    fig.tight_layout()
    save_metastudy_figure(fig, path)


def write_response_constraint_coverage(frame: pd.DataFrame, path: Path, *, primary_reduction_id: str) -> None:
    required = {
        "reduction_id",
        "selection_view_id",
        "positive_response_count",
        "zero_constraint_feasible_count",
        "n",
    }
    require_columns(frame, required, context="response constraint coverage plot")
    work = frame.loc[frame["reduction_id"].eq(primary_reduction_id)].copy()
    if work.empty:
        raise ValueError(f"constraint coverage lacks primary reduction {primary_reduction_id!r}.")
    work = work.set_index("selection_view_id").reindex(
        [value for value in _TARGET_VIEW_ORDER if value in set(work["selection_view_id"])]
    )
    target_view_ids = work.index.astype(str).tolist()
    x = np.arange(len(target_view_ids))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    response_bars = ax.bar(
        x - width / 2.0,
        work["positive_response_count"],
        width,
        label="Response ordering: min ON r > max OFF r",
        zorder=3,
    )
    feasible_bars = ax.bar(
        x + width / 2.0,
        work["zero_constraint_feasible_count"],
        width,
        label="Ordering plus ON/OFF fluorescence constraints",
        zorder=3,
    )
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    ax.set_xticks(x, [_TARGET_VIEW_MASK_LABELS[value] for value in target_view_ids])
    ax.set_ylabel(f"Observed designs (of {int(work['n'].max())})")
    ax.set_xlabel("Target mask")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.7, zorder=0)
    ax.bar_label(response_bars, padding=2, fontsize=10)
    ax.bar_label(feasible_bars, padding=2, fontsize=10)
    ax.legend(frameon=False, loc="upper right", fontsize=11)
    fig.tight_layout()
    save_metastudy_figure(fig, path)


def write_response_uncertainty_sources(frame: pd.DataFrame, path: Path) -> None:
    required = {
        "selection_view_id",
        "response_separation__bootstrap_sd",
        "response_separation__event_half_range",
        "on_magnitude_floor__bootstrap_sd",
        "on_magnitude_floor__event_half_range",
        "off_magnitude_ceiling__bootstrap_sd",
        "off_magnitude_ceiling__event_half_range",
    }
    require_columns(frame, required, context="response uncertainty plot")
    medians = frame.groupby("selection_view_id", sort=True)[list(required - {"selection_view_id"})].median()
    records: list[dict[str, object]] = []
    for selection_view_id, row in medians.iterrows():
        for component in ("response_separation", "on_magnitude_floor", "off_magnitude_ceiling"):
            records.append(
                {
                    "selection_view_id": str(selection_view_id),
                    "component": {
                        "response_separation": "Response separation",
                        "on_magnitude_floor": "ON fluorescence",
                        "off_magnitude_ceiling": "OFF fluorescence",
                    }[component],
                    "well_resampling": float(row[f"{component}__bootstrap_sd"]),
                    "event": float(row[f"{component}__event_half_range"]),
                }
            )
    work = pd.DataFrame.from_records(records)
    labels = [
        f"{_TARGET_VIEW_LABELS[str(row.selection_view_id)]}\n{row.component}" for row in work.itertuples(index=False)
    ]
    x = np.arange(len(work))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    ax.bar(x - width / 2.0, work["well_resampling"], width, label="Well-resampling bootstrap SD")
    ax.bar(x + width / 2.0, work["event"], width, label="Maximum event-bound deviation")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Uncertainty in log2(YFP / CFP) response\nor pDual-10-relative fluorescence")
    ax.set_title("Median assay uncertainty by source and metric component")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_metastudy_figure(fig, path)


def write_repeated_design_agreement(frame: pd.DataFrame, path: Path) -> None:
    value_columns = [f"{prefix}{corner}__range" for prefix in ("r", "b") for corner in ("00", "10", "01", "11")]
    require_columns(frame, {"design_id", *value_columns}, context="repeated-design agreement plot")
    work = frame.sort_values("maximum_channel_range", ascending=False, kind="mergesort").reset_index(drop=True)
    panel_count = 2 if len(work) > 8 else 1
    row_indices = np.array_split(np.arange(len(work)), panel_count)
    panel_rows = max(len(indices) for indices in row_indices)
    global_max = max(float(np.nanmax(work.loc[:, value_columns].to_numpy(dtype=float))), 1.0e-12)
    fig, axes = plt.subplots(
        1,
        panel_count,
        figsize=(11.8 if panel_count == 2 else 7.2, max(5.4, 0.55 * panel_rows + 2.4)),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for ax, indices in zip(axes[0], row_indices, strict=True):
        rows = work.iloc[indices]
        values = rows.loc[:, value_columns].to_numpy(dtype=float)
        image = ax.imshow(values, cmap="magma", vmin=0.0, vmax=global_max, aspect="equal")
        _label_repeated_design_axis(ax, rows=rows, value_columns=value_columns)
        for row, column in np.ndindex(values.shape):
            ax.text(
                column,
                row,
                f"{values[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color=contrast_text_color(image, values[row, column]),
            )
    if image is None:
        raise ValueError("repeated-design agreement plot requires at least one row.")
    fig.supxlabel("Condition-specific handoff field")
    fig.supylabel("Reader design")
    colorbar = fig.colorbar(image, ax=axes[0].tolist(), fraction=0.025, pad=0.03)
    colorbar.set_label("Cross-experiment range (log2 units)")
    save_metastudy_figure(fig, path)


def _label_repeated_design_axis(
    axis: plt.Axes,
    *,
    rows: pd.DataFrame,
    value_columns: list[str],
) -> None:
    axis.set_xticks(
        np.arange(len(value_columns)),
        [f"{value[:3]}\n{_STATE_TICK_LABELS[value[1:3]]}" for value in value_columns],
        rotation=90,
        ha="right",
        fontsize=10,
    )
    axis.set_yticks(np.arange(len(rows)), rows["design_id"].astype(str), fontsize=11)
    axis.axvline(3.5, color="#d1d5db", linewidth=0.9, zorder=5)
    add_column_group_brackets(
        axis,
        [
            (-0.45, 3.45, "Response\nlog2(YFP / CFP)"),
            (3.55, 7.45, "Relative fluorescence\nlog2(YFP / OD600) vs pDual-10"),
        ],
    )


__all__ = [
    "write_reader_event_intervals",
    "write_response_constraint_coverage",
    "write_response_separation_stability",
    "write_response_uncertainty_sources",
    "write_repeated_design_agreement",
]
