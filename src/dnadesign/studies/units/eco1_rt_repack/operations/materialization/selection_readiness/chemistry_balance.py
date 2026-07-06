"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/chemistry_balance.py

Near-DNA/RNA chemistry-balance visual for Eco1 RT panel selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.colors import TwoSlopeNorm

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.rendering import (
    LABEL_SIZE,
    TITLE_SIZE,
    save_accessible_svg,
)

from .plot_support import class_label, matrix_text_color, ordered_panel_rows, plot_row, short_candidate
from .review_axis_contracts import (
    NA_FACING_CHARGE_FIELD,
    NA_FACING_CHEMISTRY_METRICS,
    NA_FACING_CHEMISTRY_REQUIRED_FIELDS,
)
from .visual_inventory import SELECTION_PLOT_PLAIN_TITLES

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def build_na_facing_chemistry_balance_matrix(
    *,
    panel_rows: list[dict[str, object]],
    triage_rows: list[dict[str, object]],
) -> tuple[list[str], list[int], list[str], list[list[int]]]:
    """Return selected-candidate chemistry changes near retained DNA/RNA or thumb-track."""

    ordered_panel = ordered_panel_rows(panel_rows)
    triage_by_id = {str(row["candidate_id"]): row for row in triage_rows if row.get("candidate_id")}
    row_labels: list[str] = []
    charge_delta: list[int] = []
    matrix: list[list[int]] = []
    for panel_row in ordered_panel:
        candidate_id = str(panel_row["candidate_id"])
        triage_row = triage_by_id.get(candidate_id)
        if triage_row is None:
            raise ValueError(f"Selection panel references missing triage row: {candidate_id}")
        _require_na_facing_chemistry_fields(candidate_id=candidate_id, triage_row=triage_row)
        row_labels.append(f"{class_label(str(panel_row['design_class_id']))}  {short_candidate(candidate_id)}")
        charge_delta.append(int(triage_row[NA_FACING_CHARGE_FIELD]))
        matrix.append([int(triage_row[metric.field]) for metric in NA_FACING_CHEMISTRY_METRICS])
    return row_labels, charge_delta, [metric.label for metric in NA_FACING_CHEMISTRY_METRICS], matrix


def write_na_facing_chemistry_balance_plot(
    plot_root: Path,
    *,
    panel_rows: list[dict[str, object]],
    triage_rows: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    title = SELECTION_PLOT_PLAIN_TITLES["selection_na_facing_chemistry_balance"]
    row_labels, charge_delta, metric_labels, matrix = build_na_facing_chemistry_balance_matrix(
        panel_rows=panel_rows,
        triage_rows=triage_rows,
    )
    if not matrix:
        raise ValueError("near-DNA/RNA chemistry-balance plot requires selected candidates")
    fig = plt.figure(figsize=(7.8, 7.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 4.2], wspace=0.1)
    ax_charge = fig.add_subplot(grid[0, 0])
    ax_counts = fig.add_subplot(grid[0, 1], sharey=ax_charge)

    max_abs_charge = max(max((abs(value) for value in charge_delta), default=0), 1)
    ax_charge.imshow(
        [[value] for value in charge_delta],
        aspect="equal",
        interpolation="nearest",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-max_abs_charge, vcenter=0, vmax=max_abs_charge),
    )
    count_max = max((max(values) for values in matrix), default=0)
    count_image = ax_counts.imshow(
        matrix,
        aspect="equal",
        interpolation="nearest",
        cmap="YlGnBu",
        vmin=0,
        vmax=max(count_max, 1),
    )

    ax_charge.set_yticks(list(range(len(row_labels))))
    ax_charge.set_yticklabels(row_labels, fontsize=LABEL_SIZE - 0.5)
    ax_charge.set_xticks([0])
    ax_charge.set_xticklabels(["Charge change"], fontsize=LABEL_SIZE - 1, rotation=24, ha="right")
    ax_counts.set_xticks(list(range(len(metric_labels))))
    ax_counts.set_xticklabels(metric_labels, fontsize=LABEL_SIZE - 1, rotation=24, ha="right")
    ax_counts.tick_params(axis="y", left=False, labelleft=False)

    for row_index, value in enumerate(charge_delta):
        ax_charge.text(
            0,
            row_index,
            f"{value:+d}",
            ha="center",
            va="center",
            fontsize=9.4,
            color=matrix_text_color(float(abs(value)), max_value=float(max_abs_charge)),
        )
    for row_index, values in enumerate(matrix):
        for col_index, value in enumerate(values):
            ax_counts.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                fontsize=9.4,
                color=matrix_text_color(float(value), max_value=float(max(count_max, 1))),
            )

    ax_charge.set_title(title, fontsize=TITLE_SIZE, pad=12, loc="left")
    for axis in (ax_charge, ax_counts):
        axis.tick_params(axis="both", length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
    count_bar = fig.colorbar(count_image, ax=ax_counts, shrink=0.72, pad=0.02)
    count_bar.set_label("Substitution count", fontsize=10.5)
    count_bar.ax.tick_params(labelsize=9.5)
    fig.subplots_adjust(left=0.32, right=0.94, top=0.88, bottom=0.26)

    path = plot_root / "selection_na_facing_chemistry_balance.svg"
    alt = (
        "Heatmap of selected Eco1 RT candidates showing charge delta and residue-class substitution counts "
        "for mutations near retained DNA/RNA or thumb-track positions."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_na_facing_chemistry_balance",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Separates the near retained DNA/RNA or thumb-track chemistry fields into charge change, basic gain, "
            "basic loss, acidic gain, and proline/glycine gain for the selected panel."
        ),
        interpretation_limit=(
            "Chemistry changes are review risks and context only. They do not establish activity, processivity, "
            "strand displacement, or assay readiness."
        ),
        render_mode="wide_visual",
    )


def _require_na_facing_chemistry_fields(*, candidate_id: str, triage_row: dict[str, object]) -> None:
    missing = [field for field in NA_FACING_CHEMISTRY_REQUIRED_FIELDS if triage_row.get(field) is None]
    if missing:
        raise ValueError(
            f"Selected triage row is missing near-DNA/RNA chemistry fields for {candidate_id}: {', '.join(missing)}"
        )
