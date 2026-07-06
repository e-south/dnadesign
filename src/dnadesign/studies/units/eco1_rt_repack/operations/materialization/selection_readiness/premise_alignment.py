"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/premise_alignment.py

Selected-panel premise-alignment visual for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.rendering import (
    LABEL_SIZE,
    LEGEND_SIZE,
    OKABE_ITO,
    TITLE_SIZE,
    save_accessible_svg,
)

from .plot_support import class_label, ordered_panel_rows, plot_row, short_candidate
from .visual_inventory import SELECTION_PLOT_PLAIN_TITLES

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class PremiseCell:
    """One displayed value in the selected-panel premise matrix."""

    text: str
    kind: str


_COLUMN_LABELS = [
    "Core/direct edits",
    "NA annulus edits",
    "Thumb-track edits",
    "Distal edits",
    "Chemistry warnings",
    "Fold gate",
    "ESMC/SAE",
]
_KIND_TO_CODE = {
    "protected": 0,
    "review": 1,
    "context": 2,
    "warning": 3,
    "blocked": 4,
}
_KIND_COLORS = [
    "#f7f5ef",
    OKABE_ITO["blue"],
    OKABE_ITO["green"],
    OKABE_ITO["orange"],
    OKABE_ITO["vermillion"],
]


def build_premise_alignment_matrix(
    *,
    panel_rows: list[dict[str, object]],
    triage_rows: list[dict[str, object]],
) -> tuple[list[str], list[str], list[list[PremiseCell]]]:
    """Return selected-candidate rows for the core premise review matrix."""

    triage_by_id = {str(row["candidate_id"]): row for row in triage_rows if row.get("candidate_id")}
    row_labels: list[str] = []
    matrix: list[list[PremiseCell]] = []
    for panel_row in ordered_panel_rows(panel_rows):
        candidate_id = str(panel_row["candidate_id"])
        triage_row = triage_by_id.get(candidate_id)
        if triage_row is None:
            raise ValueError(f"Selection panel references missing triage row: {candidate_id}")
        row_labels.append(f"{class_label(str(panel_row['design_class_id']))}  {short_candidate(candidate_id)}")
        thumb_count = _int_field(triage_row, "thumb_contact_track_mutation_count")
        na_annulus_count = max(_int_field(triage_row, "nucleic_acid_facing_mutation_count") - thumb_count, 0)
        core_direct_count = _int_field(triage_row, "catalytic_or_direct_contact_mutation_count")
        chemistry_warnings = _int_field(triage_row, "nucleic_acid_facing_chemistry_warning_count")
        matrix.append(
            [
                PremiseCell(str(core_direct_count), "protected" if core_direct_count == 0 else "blocked"),
                PremiseCell(str(na_annulus_count), "review" if na_annulus_count else "protected"),
                PremiseCell(str(thumb_count), "review" if thumb_count else "protected"),
                PremiseCell(str(_int_field(triage_row, "distal_scaffold_mutation_count")), "context"),
                PremiseCell(str(chemistry_warnings), "protected" if chemistry_warnings == 0 else "warning"),
                PremiseCell(_fold_gate_text(triage_row), _fold_gate_kind(triage_row)),
                PremiseCell("review", "review"),
            ]
        )
    return list(_COLUMN_LABELS), row_labels, matrix


def write_premise_alignment_plot(
    plot_root: Path,
    *,
    panel_rows: list[dict[str, object]],
    triage_rows: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    """Write the selected-six premise-alignment SVG."""

    title = SELECTION_PLOT_PLAIN_TITLES["selection_premise_alignment"]
    column_labels, row_labels, matrix = build_premise_alignment_matrix(panel_rows=panel_rows, triage_rows=triage_rows)
    if not matrix:
        raise ValueError("premise-alignment plot requires selected candidates")
    codes = [[_KIND_TO_CODE[cell.kind] for cell in row] for row in matrix]
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    cmap = ListedColormap(_KIND_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    ax.imshow(codes, aspect="equal", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_xticks(list(range(len(column_labels))))
    ax.set_xticklabels(column_labels, fontsize=LABEL_SIZE - 1, rotation=24, ha="right")
    ax.set_yticks(list(range(len(row_labels))))
    ax.set_yticklabels(row_labels, fontsize=LABEL_SIZE - 0.5)
    for row_index, values in enumerate(matrix):
        for col_index, cell in enumerate(values):
            ax.text(
                col_index,
                row_index,
                cell.text,
                ha="center",
                va="center",
                fontsize=9.3,
                color="#ffffff" if cell.kind in {"review", "context", "warning", "blocked"} else "#24292f",
            )
    ax.set_title(title, fontsize=TITLE_SIZE, pad=12)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [
        Patch(facecolor="#f7f5ef", edgecolor="#d8dee4", label="Protected or zero"),
        Patch(facecolor=OKABE_ITO["blue"], edgecolor="#ffffff", label="Review context"),
        Patch(facecolor=OKABE_ITO["green"], edgecolor="#ffffff", label="Distal context"),
        Patch(facecolor=OKABE_ITO["orange"], edgecolor="#ffffff", label="Chemistry warning"),
        Patch(facecolor=OKABE_ITO["vermillion"], edgecolor="#ffffff", label="Blocked"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
        fontsize=LEGEND_SIZE,
        columnspacing=1.0,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.31, right=0.97, top=0.88, bottom=0.27)
    path = plot_root / "selection_premise_alignment.svg"
    alt = (
        "Selected-six matrix showing catalytic or direct-contact edit counts, near-DNA/RNA annulus edit counts, "
        "thumb-track edit counts, distal edit counts, near-DNA/RNA chemistry warnings, fold gate status, and "
        "ESMC/SAE review-only status."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_premise_alignment",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Summarizes the selected panel against the study premise: protect catalytic and direct-contact positions, "
            "review peripheral near-DNA/RNA changes, treat distal edits as scaffold context, and keep ESMC/SAE as "
            "annotations."
        ),
        interpretation_limit=(
            "This matrix is a review checklist. It does not establish activity, processivity, strand displacement, "
            "or assay readiness."
        ),
        render_mode="wide_visual",
    )


def _int_field(row: dict[str, object], field: str) -> int:
    return int(row.get(field) or 0)


def _fold_gate_text(row: dict[str, object]) -> str:
    review_class = str(row.get("fold_review_class") or "")
    if review_class == "strong_fold_preserved":
        return "strong"
    if review_class == "good_fold_preserved":
        return "good"
    if review_class == "review_band":
        return "reserve"
    if review_class == "low_confidence":
        return "blocked"
    return review_class.replace("_", " ") or "missing"


def _fold_gate_kind(row: dict[str, object]) -> str:
    if str(row.get("hard_gate_status") or "") == "eligible":
        return "context"
    if str(row.get("hard_gate_status") or "") == "needs_review":
        return "warning"
    return "blocked"
