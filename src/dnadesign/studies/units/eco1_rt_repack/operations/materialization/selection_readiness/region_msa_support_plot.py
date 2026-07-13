"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/region_msa_support_plot.py

Region-wise MSA support plot for Eco1 RT panel selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.shared.rendering import (
    LABEL_SIZE,
    TITLE_SIZE,
    save_accessible_svg,
)

from .plot_support import matrix_text_color, ordered_panel_rows, plot_row, policy_label, short_candidate
from .region_msa_support import REGION_MSA_SUPPORT_REGION_IDS
from .visual_inventory import SELECTION_PLOT_PLAIN_TITLES

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def build_selected_region_msa_support_matrix(
    *,
    panel_rows: list[dict[str, object]],
    region_msa_support_rows: list[dict[str, object]],
) -> tuple[list[str], list[str], list[list[float | None]], list[list[int]]]:
    """Return selected-candidate MSA support by mutation region."""

    rows_by_candidate_region = {
        (str(row["candidate_id"]), str(row["region_id"])): row
        for row in region_msa_support_rows
        if row.get("candidate_id") and row.get("region_id")
    }
    labels_by_region = {
        str(row["region_id"]): str(row.get("region_label") or row["region_id"]) for row in region_msa_support_rows
    }
    region_labels = [
        labels_by_region.get(region_id, region_id.replace("_", " ")) for region_id in REGION_MSA_SUPPORT_REGION_IDS
    ]
    row_labels: list[str] = []
    fraction_matrix: list[list[float | None]] = []
    unobserved_matrix: list[list[int]] = []
    for panel_row in ordered_panel_rows(panel_rows):
        candidate_id = str(panel_row["candidate_id"])
        row_labels.append(f"{policy_label(str(panel_row['policy_id']))}  {short_candidate(candidate_id)}")
        fractions: list[float | None] = []
        unobserved: list[int] = []
        for region_id in REGION_MSA_SUPPORT_REGION_IDS:
            row = rows_by_candidate_region.get((candidate_id, region_id))
            if row is None:
                fractions.append(None)
                unobserved.append(0)
                continue
            value = row.get("alt_observed_fraction")
            fractions.append(None if value is None else float(value))
            unobserved.append(int(row.get("unobserved_mutation_count") or 0))
        fraction_matrix.append(fractions)
        unobserved_matrix.append(unobserved)
    if not fraction_matrix:
        raise ValueError("region-wise MSA support plot requires selected candidates")
    return region_labels, row_labels, fraction_matrix, unobserved_matrix


def write_regionwise_msa_support_plot(
    plot_root: Path,
    *,
    panel_rows: list[dict[str, object]],
    region_msa_support_rows: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    """Write selected-candidate region-wise MSA support heatmap."""

    title = SELECTION_PLOT_PLAIN_TITLES["selection_regionwise_msa_support"]
    region_labels, row_labels, fractions, unobserved = build_selected_region_msa_support_matrix(
        panel_rows=panel_rows,
        region_msa_support_rows=region_msa_support_rows,
    )
    plot_values = np.asarray(
        [[np.nan if value is None else value for value in row] for row in fractions],
        dtype=float,
    )
    masked_values = np.ma.masked_invalid(plot_values)
    fig, ax = plt.subplots(figsize=(8.6, 11.7))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#d0d7de")
    ax.imshow(masked_values, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_yticks(list(range(len(row_labels))))
    ax.set_yticklabels(row_labels, fontsize=LABEL_SIZE - 0.5)
    ax.set_xticks(list(range(len(region_labels))))
    ax.set_xticklabels(region_labels, fontsize=LABEL_SIZE - 1, rotation=24, ha="right")
    for row_index, values in enumerate(fractions):
        for col_index, value in enumerate(values):
            if value is None:
                text = "no edits"
                color = "#24292f"
            else:
                missing_count = unobserved[row_index][col_index]
                if missing_count == 0:
                    text = f"{value:.0%} observed\nall edits seen"
                else:
                    suffix = "edit" if missing_count == 1 else "edits"
                    text = f"{value:.0%} observed\n{missing_count} {suffix} not seen"
                color = matrix_text_color(value, max_value=1.0)
            ax.text(col_index, row_index, text, ha="center", va="center", fontsize=8.2, color=color)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=12)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.31, right=0.96, top=0.90, bottom=0.27)
    path = plot_root / "selection_regionwise_msa_support.svg"
    alt = (
        "Heatmap of selected Eco1 RT candidates showing the fraction of designed substitutions observed in "
        "the clade-9 natural-sequence alignment "
        "within catalytic/direct-contact, near retained DNA/RNA, thumb-contact, C-terminal primer-RNA recognition, "
        "and distal regions."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_regionwise_msa_support",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Shows the fraction of selected substitutions in each review region that also occur in the clade-9 "
            "alignment. The C-terminal primer-RNA recognition region is an overlapping context. Cells with no "
            "mutations are labeled as no edits rather than scored. The peripheral alphabet limits substitutions "
            "to observed alternatives; this plot audits that generation contract."
        ),
        interpretation_limit=(
            "Observed substitutions are a sequence prior, not functional proof. Region-wise MSA support is not a "
            "composite selection score."
        ),
        render_mode="wide_visual",
    )


__all__ = [
    "build_selected_region_msa_support_matrix",
    "write_regionwise_msa_support_plot",
]
