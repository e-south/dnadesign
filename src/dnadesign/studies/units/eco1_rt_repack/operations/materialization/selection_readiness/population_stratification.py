"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/population_stratification.py

Full-population stratification plot for Eco1 selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.rendering import (
    LABEL_SIZE,
    LEGEND_SIZE,
    OKABE_ITO,
    TITLE_SIZE,
    save_accessible_svg,
    style_open_axes,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.plot_support import (
    class_label,
    plot_row,
    short_candidate,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_STATUS_ORDER = ("eligible", "needs_review", "ineligible", "missing_inputs")
_STATUS_COLORS = {
    "eligible": OKABE_ITO["green"],
    "needs_review": OKABE_ITO["orange"],
    "ineligible": "#8c959f",
    "missing_inputs": OKABE_ITO["vermillion"],
}
_STATUS_LABELS = {
    "eligible": "Eligible",
    "needs_review": "Manual reserve",
    "ineligible": "Excluded",
    "missing_inputs": "Missing input",
}


def write_population_stratification_plot(
    plot_root: Path,
    triage_rows: list[dict[str, object]],
    panel_rows: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    """Write the full-candidate-population plot with selected panel rows highlighted."""

    title = "Selected Eco1 candidates are shown in the context of the full candidate population"
    selected_by_id = {str(row["candidate_id"]): row for row in panel_rows}
    plot_rows = [
        row
        for row in triage_rows
        if row.get("selection_support_alt_observed_fraction") is not None
        and row.get("nucleic_acid_facing_mutation_count") is not None
    ]
    if not plot_rows:
        raise ValueError("population stratification plot requires triage rows with selection-support axes")

    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    for status in _STATUS_ORDER:
        rows_for_status = [row for row in plot_rows if str(row.get("hard_gate_status") or "") == status]
        if not rows_for_status:
            continue
        ax.scatter(
            [float(row["selection_support_alt_observed_fraction"]) for row in rows_for_status],
            [int(row["nucleic_acid_facing_mutation_count"]) for row in rows_for_status],
            s=[42 + (int(row.get("nucleic_acid_facing_chemistry_warning_count") or 0) * 10) for row in rows_for_status],
            c=_STATUS_COLORS[status],
            alpha=0.54 if status == "eligible" else 0.4,
            edgecolors="#ffffff",
            linewidths=0.35,
            label=_STATUS_LABELS[status],
            zorder=2,
        )

    selected_rows = [row for row in plot_rows if str(row.get("candidate_id") or "") in selected_by_id]
    if not selected_rows:
        raise ValueError("population stratification plot requires selected candidates in triage rows")
    selected_x = [float(row["selection_support_alt_observed_fraction"]) for row in selected_rows]
    selected_y = [int(row["nucleic_acid_facing_mutation_count"]) for row in selected_rows]
    ax.scatter(
        selected_x,
        selected_y,
        s=160,
        facecolors="none",
        edgecolors="#24292f",
        linewidths=1.45,
        marker="o",
        label="Selected panel",
        zorder=4,
    )
    for row, x_value, y_value in zip(selected_rows, selected_x, selected_y, strict=True):
        panel_row = selected_by_id[str(row["candidate_id"])]
        label = f"{class_label(str(panel_row['design_class_id']))}: {short_candidate(str(row['candidate_id']))}"
        ax.text(
            min(x_value + 0.016, 1.01),
            y_value + 0.08,
            label,
            fontsize=9.3,
            color="#24292f",
            ha="left",
            va="bottom",
            zorder=5,
        )

    ax.set_xlim(-0.02, 1.06)
    y_max = max(int(row["nucleic_acid_facing_mutation_count"]) for row in plot_rows)
    ax.set_ylim(-0.7, y_max + 2.0)
    ax.set_xlabel("Designed substitutions observed in the selected MSA denominator", fontsize=LABEL_SIZE)
    ax.set_ylabel("Nucleic-acid-facing designed substitutions", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=12)
    style_open_axes(ax)
    ax.grid(color="#d0d7de", alpha=0.45, linewidth=0.7)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=3,
        frameon=False,
        fontsize=LEGEND_SIZE,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.25)
    path = plot_root / "selection_population_stratification.svg"
    alt = (
        "Scatter plot of the full candidate population. X position is MSA-observed designed-substitution "
        "fraction, y position is nucleic-acid-facing mutation count, point color marks hard-gate status, "
        "and outlined markers identify the six selected Eco1 candidates."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_population_stratification",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Places the six selected candidates against the full candidate population using the same "
            "review axes used for panel selection."
        ),
        interpretation_limit=(
            "The plot is a stratification aid for review. It is not a combined score, activity measurement, "
            "or claim of improved strand displacement."
        ),
        render_mode="wide_visual",
    )
