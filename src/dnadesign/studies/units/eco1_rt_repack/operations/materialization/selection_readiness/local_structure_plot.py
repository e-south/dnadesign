"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/local_structure_plot.py

Local-structure review visual for Eco1 RT selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.rendering import (
    LABEL_SIZE,
    TITLE_SIZE,
    save_accessible_svg,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.local_structure import (
    LOCAL_STRUCTURE_REGION_IDS,
    LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_ID,
    LOCAL_STRUCTURE_RMSD_THRESHOLDS_ANGSTROM,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.plot_support import (
    class_label,
    matrix_text_color,
    ordered_panel_rows,
    plot_row,
    short_candidate,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.visual_inventory import (
    SELECTION_PLOT_PLAIN_TITLES,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def build_selected_local_structure_matrix(
    *,
    panel_rows: list[dict[str, object]],
    local_structure_rows: list[dict[str, object]],
) -> tuple[list[str], list[str], list[list[float | None]], list[list[str]]]:
    """Return selected-candidate local-structure metrics by region."""

    rows_by_candidate_region = {
        (str(row["candidate_id"]), str(row["region_id"])): row
        for row in local_structure_rows
        if row.get("candidate_id") and row.get("region_id")
    }
    labels_by_region = {
        str(row["region_id"]): str(row.get("region_label") or row["region_id"]) for row in local_structure_rows
    }
    region_labels = [
        labels_by_region.get(region_id, region_id.replace("_", " ")) for region_id in LOCAL_STRUCTURE_REGION_IDS
    ]
    row_labels: list[str] = []
    matrix: list[list[float | None]] = []
    status_matrix: list[list[str]] = []
    for panel_row in ordered_panel_rows(panel_rows):
        candidate_id = str(panel_row["candidate_id"])
        row_labels.append(f"{class_label(str(panel_row['design_class_id']))}  {short_candidate(candidate_id)}")
        values: list[float | None] = []
        statuses: list[str] = []
        for region_id in LOCAL_STRUCTURE_REGION_IDS:
            row = rows_by_candidate_region.get((candidate_id, region_id))
            if row is None:
                values.append(None)
                statuses.append("missing_metric")
                continue
            value = row.get("local_ca_rmsd_angstrom")
            values.append(None if value is None else float(value))
            statuses.append(str(row.get("status") or "missing_status"))
        matrix.append(values)
        status_matrix.append(statuses)
    if not matrix:
        raise ValueError("local-structure plot requires selected candidates")
    return region_labels, row_labels, matrix, status_matrix


def _region_label_with_threshold(*, region_id: str, label: str) -> str:
    threshold = LOCAL_STRUCTURE_RMSD_THRESHOLDS_ANGSTROM[region_id]
    return f"{label}\n<= {threshold:.2f} A"


def write_local_structure_by_region_plot(
    plot_root: Path,
    *,
    panel_rows: list[dict[str, object]],
    local_structure_rows: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    """Write selected-candidate local C-alpha RMSD heatmap."""

    title = SELECTION_PLOT_PLAIN_TITLES["selection_local_structure_by_region"]
    region_labels, row_labels, matrix, status_matrix = build_selected_local_structure_matrix(
        panel_rows=panel_rows,
        local_structure_rows=local_structure_rows,
    )
    region_labels = [
        _region_label_with_threshold(region_id=region_id, label=label)
        for region_id, label in zip(LOCAL_STRUCTURE_REGION_IDS, region_labels, strict=True)
    ]
    numeric_values = [value for row in matrix for value in row if value is not None]
    max_value = max(numeric_values, default=1.0)
    plot_values = np.asarray([[np.nan if value is None else value for value in row] for row in matrix], dtype=float)
    masked_values = np.ma.masked_invalid(plot_values)
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#d0d7de")
    image = ax.imshow(masked_values, aspect="equal", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=max_value)
    ax.set_yticks(list(range(len(row_labels))))
    ax.set_yticklabels(row_labels, fontsize=LABEL_SIZE - 0.5)
    ax.set_xticks(list(range(len(region_labels))))
    ax.set_xticklabels(region_labels, fontsize=LABEL_SIZE - 1.8, rotation=25, ha="right")
    for row_index, values in enumerate(matrix):
        for col_index, value in enumerate(values):
            if value is None:
                text = "NA"
                color = "#24292f"
            else:
                text = f"{value:.2f}"
                color = matrix_text_color(value, max_value=max_value)
            ax.text(col_index, row_index, text, ha="center", va="center", fontsize=8.6, color=color)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=12)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.02)
    cbar.set_label("Local C-alpha RMSD (A)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=0.3, right=0.94, top=0.88, bottom=0.28)
    path = plot_root / "selection_local_structure_by_region.svg"
    unavailable_statuses = sorted({status for row in status_matrix for status in row if status != "available"})
    alt = (
        "Heatmap of selected Eco1 RT candidates by local C-alpha RMSD in motif, thumb-track, annulus, and distal "
        "regions after one global mapped C-alpha fit. Column labels include the local RMSD threshold. Unavailable "
        "cells are labeled NA."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_local_structure_by_region",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Shows local backbone shifts by RT review region after a single global mapped C-alpha alignment. "
            "Every selected row must have all declared local-structure metrics and stay within the local RMSD "
            "threshold shown for each region. "
            f"Unavailable statuses: {', '.join(unavailable_statuses) if unavailable_statuses else 'none'}."
        ),
        interpretation_limit=(
            "Local C-alpha RMSD is a structural review metric, not an activity, processivity, strand-displacement, "
            "or assay-readiness measurement."
        ),
        render_mode="wide_visual",
    )


def write_local_structure_stratification_plot(
    plot_root: Path,
    *,
    triage_rows: list[dict[str, object]],
    panel_rows: list[dict[str, object]],
    local_structure_rows: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    """Write population local-RMSD threshold stratification plot."""

    title = SELECTION_PLOT_PLAIN_TITLES["selection_local_structure_stratification"]
    selected_candidates = {str(row["candidate_id"]) for row in panel_rows}
    hard_gate_by_candidate = {
        str(row["candidate_id"]): str(row.get("hard_gate_status") or "")
        for row in triage_rows
        if row.get("candidate_id")
    }
    labels_by_region = {
        str(row["region_id"]): str(row.get("region_label") or row["region_id"]) for row in local_structure_rows
    }
    values_by_region: dict[str, list[tuple[float, str, bool, str]]] = {
        region_id: [] for region_id in LOCAL_STRUCTURE_REGION_IDS
    }
    for row in local_structure_rows:
        region_id = str(row.get("region_id") or "")
        if region_id not in values_by_region or str(row.get("status") or "") != "available":
            continue
        value = row.get("local_ca_rmsd_angstrom")
        if value is None:
            continue
        candidate_id = str(row.get("candidate_id") or "")
        values_by_region[region_id].append(
            (
                float(value),
                candidate_id,
                candidate_id in selected_candidates,
                hard_gate_by_candidate.get(candidate_id, ""),
            )
        )
    region_labels = [
        labels_by_region.get(region_id, region_id.replace("_", " ")) for region_id in LOCAL_STRUCTURE_REGION_IDS
    ]
    fig, ax = plt.subplots(figsize=(9.8, 7.4))
    y_positions = np.arange(len(LOCAL_STRUCTURE_REGION_IDS), dtype=float)
    max_x = 0.0
    for y_index, region_id in enumerate(LOCAL_STRUCTURE_REGION_IDS):
        region_values = values_by_region[region_id]
        max_x = max(max_x, max([value for value, *_rest in region_values], default=0.0))
        threshold = LOCAL_STRUCTURE_RMSD_THRESHOLDS_ANGSTROM[region_id]
        max_x = max(max_x, threshold)
        all_values = [value for value, _candidate_id, selected, _status in region_values if not selected]
        selected_values = [value for value, _candidate_id, selected, _status in region_values if selected]
        if all_values:
            jitter = _deterministic_jitter(len(all_values), amplitude=0.22)
            ax.scatter(
                all_values,
                [y_positions[y_index] + offset for offset in jitter],
                s=15,
                color="#8c959f",
                alpha=0.24,
                linewidth=0,
                label="Other candidates" if y_index == 0 else None,
                zorder=2,
            )
        if selected_values:
            ax.scatter(
                selected_values,
                [y_positions[y_index]] * len(selected_values),
                s=96,
                color="#0072b2",
                edgecolor="#ffffff",
                linewidth=1.1,
                label="Selected rows" if y_index == 0 else None,
                zorder=5,
            )
        ax.plot(
            [threshold, threshold],
            [y_positions[y_index] - 0.35, y_positions[y_index] + 0.35],
            color="#d55e00",
            linewidth=2.2,
            solid_capstyle="butt",
            label="Threshold" if y_index == 0 else None,
            zorder=4,
        )
        failed = sum(value > threshold for value, *_rest in region_values)
        ax.text(
            max_x + 0.08,
            y_positions[y_index],
            f">{threshold:.2f} A: {failed} fail",
            ha="left",
            va="center",
            fontsize=9.2,
            color="#57606a",
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(region_labels, fontsize=LABEL_SIZE - 0.5)
    ax.invert_yaxis()
    ax.set_xlabel("Local C-alpha RMSD (A)", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=12)
    ax.set_xlim(left=0.0, right=max_x + 1.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=10)
    style = {"color": "#d8dee4", "linewidth": 0.7}
    ax.grid(axis="x", **style)
    ax.grid(axis="y", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.29, right=0.96, top=0.9, bottom=0.24)
    path = plot_root / "selection_local_structure_stratification.svg"
    alt = (
        "Population stratification plot for local C-alpha RMSD by RT review region. Gray points are nonselected "
        "candidates, blue points are selected panel rows, and orange markers show exploratory per-region thresholds."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_local_structure_stratification",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Shows each local-structure threshold relative to the candidate population. Rows exceeding a "
            f"region threshold fail the local-structure gate under {LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_ID}."
        ),
        interpretation_limit=(
            "These thresholds are structural preservation gates for review readiness. They do not measure RT activity, "
            "processivity, strand displacement, or assay readiness."
        ),
        render_mode="wide_visual",
    )


def _deterministic_jitter(count: int, *, amplitude: float) -> list[float]:
    if count <= 1:
        return [0.0] * count
    offsets = np.linspace(-amplitude, amplitude, num=count)
    return [float(offset) for offset in offsets]


__all__ = [
    "build_selected_local_structure_matrix",
    "write_local_structure_by_region_plot",
    "write_local_structure_stratification_plot",
]
