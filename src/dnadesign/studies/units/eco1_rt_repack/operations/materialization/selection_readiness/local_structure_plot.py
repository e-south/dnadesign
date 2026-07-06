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
    ax.set_xticklabels(region_labels, fontsize=LABEL_SIZE - 1.2, rotation=25, ha="right")
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
    cbar.set_label("Local C-alpha RMSD after global fit (A)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=0.3, right=0.94, top=0.88, bottom=0.28)
    path = plot_root / "selection_local_structure_by_region.svg"
    unavailable_statuses = sorted({status for row in status_matrix for status in row if status != "available"})
    alt = (
        "Heatmap of selected Eco1 RT candidates by local C-alpha RMSD in motif, thumb-track, annulus, and distal "
        "regions after one global mapped C-alpha fit. Unavailable cells are labeled NA."
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
            f"Unavailable statuses: {', '.join(unavailable_statuses) if unavailable_statuses else 'none'}."
        ),
        interpretation_limit=(
            "Local C-alpha RMSD is structural review evidence only. It is not an activity, processivity, "
            "strand-displacement, or assay-readiness measurement."
        ),
        render_mode="wide_visual",
    )


__all__ = [
    "build_selected_local_structure_matrix",
    "write_local_structure_by_region_plot",
]
