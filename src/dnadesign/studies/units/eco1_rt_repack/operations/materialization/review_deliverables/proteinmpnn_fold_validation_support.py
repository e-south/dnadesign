"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/proteinmpnn_fold_validation_support.py

Plot support helpers for Eco1 ProteinMPNN fold-validation visuals.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from matplotlib.lines import Line2D

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import (
    ALL_SPECS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.rendering import (
    LEGEND_SIZE,
    OKABE_ITO,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.plot_support import (
    class_label,
)

TEMPERATURE_COLORS = {
    0.1: OKABE_ITO["green"],
    0.3: OKABE_ITO["orange"],
}
DESIGN_CLASS_COLORS = (
    OKABE_ITO["blue"],
    OKABE_ITO["orange"],
    OKABE_ITO["green"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["purple"],
    OKABE_ITO["sky"],
)
TEMPERATURE_MARKERS = ("o", "s", "^", "D", "P", "X")


def join_expanded_fold_rows(
    *,
    candidate_pool_path: Path,
    foldcheck_ranking_path: Path,
    selection_panel_table_path: Path,
) -> list[dict[str, Any]]:
    candidate_rows = pq.read_table(
        candidate_pool_path,
        columns=["candidate_id", "design_class_id", "temperature", "seed"],
    ).to_pylist()
    ranking_rows = pq.read_table(
        foldcheck_ranking_path,
        columns=["candidate_id", "wt_runtime_ca_rmsd", "plddt", "review_rank", "review_class"],
    ).to_pylist()
    selected_ids = {
        str(row["candidate_id"])
        for row in pq.read_table(selection_panel_table_path, columns=["candidate_id"]).to_pylist()
    }
    candidate_by_id = {str(row["candidate_id"]): row for row in candidate_rows}
    joined_rows: list[dict[str, Any]] = []
    for ranking in ranking_rows:
        candidate_id = str(ranking["candidate_id"])
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            continue
        joined_rows.append(
            {
                "candidate_id": candidate_id,
                "design_class_id": str(candidate["design_class_id"]),
                "temperature": float(candidate["temperature"]),
                "seed": int(candidate["seed"]),
                "wt_runtime_ca_rmsd": float(ranking["wt_runtime_ca_rmsd"]),
                "plddt": float(ranking["plddt"]),
                "review_rank": int(ranking["review_rank"]),
                "review_class": str(ranking["review_class"]),
                "selected_for_panel": candidate_id in selected_ids,
            }
        )
    return sorted(joined_rows, key=lambda row: (row["review_rank"], row["candidate_id"]))


def add_temperature_legend(fig: Any, temperatures: list[float]) -> None:
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=temperature_color(temperature),
            markeredgecolor="#ffffff",
            label=f"Temperature {temperature:g}",
        )
        for temperature in temperatures
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=LEGEND_SIZE,
        title="Sampling temperature",
        title_fontsize=LEGEND_SIZE,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.028),
        ncol=len(legend_handles),
    )


def add_expanded_fold_legend(fig: Any, *, class_order: list[str], temperatures: list[float]) -> None:
    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=design_class_color(class_id),
            markeredgecolor="#ffffff",
            label=class_label(class_id),
        )
        for class_id in class_order
    ]
    temperature_handles = [
        Line2D(
            [0],
            [0],
            marker=temperature_marker(temperature, temperatures),
            color="#333333",
            linestyle="none",
            markerfacecolor="#333333",
            label=f"Temperature {temperature:g}",
        )
        for temperature in temperatures
    ]
    selected_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="none",
        markeredgecolor="#111111",
        markeredgewidth=1.4,
        label="Selected panel",
    )
    fig.legend(
        handles=[*class_handles, *temperature_handles, selected_handle],
        frameon=False,
        fontsize=LEGEND_SIZE,
        title="Design class / Sampling temperature",
        title_fontsize=LEGEND_SIZE,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.025),
        ncol=3,
        columnspacing=1.05,
        handletextpad=0.45,
    )


def histogram_bins(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        center = float(values[0]) if values.size else 0.0
        return np.array([center - 0.5, center + 0.5], dtype=float)
    return np.histogram_bin_edges(values, bins="auto")


def temperature_color(temperature: float) -> str:
    return TEMPERATURE_COLORS.get(round(float(temperature), 3), "#5b7fa6")


def design_class_order(rows: list[dict[str, Any]]) -> list[str]:
    known_order = [spec.design_class_id for spec in ALL_SPECS]
    present = {str(row["design_class_id"]) for row in rows}
    ordered = [class_id for class_id in known_order if class_id in present]
    ordered.extend(sorted(present - set(known_order)))
    return ordered


def design_class_color(class_id: str) -> str:
    known_order = [spec.design_class_id for spec in ALL_SPECS]
    if class_id in known_order:
        return DESIGN_CLASS_COLORS[known_order.index(class_id) % len(DESIGN_CLASS_COLORS)]
    return "#6f7782"


def temperature_marker(temperature: float, temperatures: list[float]) -> str:
    rounded = round(float(temperature), 3)
    ordered = [round(float(value), 3) for value in temperatures]
    index = ordered.index(rounded) if rounded in ordered else 0
    return TEMPERATURE_MARKERS[index % len(TEMPERATURE_MARKERS)]


def annotate_selected_rows(ax: Any, selected_rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(selected_rows):
        ax.annotate(
            str(index + 1),
            xy=(row["wt_runtime_ca_rmsd"], row["plddt"]),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=LEGEND_SIZE - 2.5,
            color="#111111",
            ha="center",
            va="center",
            zorder=6,
            bbox={
                "boxstyle": "circle,pad=0.18",
                "facecolor": "#ffffff",
                "edgecolor": "#111111",
                "linewidth": 0.55,
                "alpha": 0.92,
            },
        )
