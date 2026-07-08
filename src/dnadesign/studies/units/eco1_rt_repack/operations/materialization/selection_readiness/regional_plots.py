"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/regional_plots.py

Region-aware Eco1 RT panel-selection visuals.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.shared.rendering import (
    LABEL_SIZE,
    LEGEND_SIZE,
    OKABE_ITO,
    TITLE_SIZE,
    save_accessible_svg,
)

from ..shared.design_class_mask_annotations import add_rt_annotation_context
from ..shared.rt_annotation_context import RTAnnotationContext
from .plot_support import (
    canonical_mutations,
    class_label,
    matrix_text_color,
    mutation_category,
    ordered_panel_rows,
    parse_mutation,
    plot_row,
    position_tick_indices,
    short_candidate,
    tie_break_trace,
)
from .review_axes import (
    C_TERMINAL_PRIMER_RNA_RECOGNITION_POSITIONS,
    DIRECT_CONTACT_DISTANCE_ANGSTROM,
    NA_FACING_DISTANCE_ANGSTROM,
    WANG_THUMB_CONTACT_TRACK_POSITIONS,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def build_regional_mutation_burden_matrix(
    *,
    panel_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    mask_residues: list[dict[str, object]],
) -> tuple[list[str], list[str], list[list[int]]]:
    """Return selected-candidate mutation counts by RT review region."""

    region_labels = [
        "Catalytic or direct contact",
        "Near retained DNA/RNA region",
        "Thumb-contact track",
        "C-terminal primer-RNA recognition region",
        "Distal scaffold",
    ]
    ordered_panel = ordered_panel_rows(panel_rows)
    candidate_by_id = {str(row["candidate_id"]): row for row in candidate_rows if row.get("candidate_id")}
    residue_by_position = {int(row["canonical_position"]): row for row in mask_residues}
    row_labels: list[str] = []
    matrix: list[list[int]] = []
    for panel_row in ordered_panel:
        candidate_id = str(panel_row["candidate_id"])
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            raise ValueError(f"Selection panel references missing candidate row: {candidate_id}")
        counts = _regional_counts_from_panel_trace(panel_row)
        if counts is None:
            counts = [0, 0, 0, 0, 0]
            for mutation in canonical_mutations(candidate.get("canonical_mutations")):
                parsed = parse_mutation(mutation)
                position = int(parsed["position"])
                counts[_regional_bucket_index(position, residue_by_position.get(position, {}))] += 1
                if position in C_TERMINAL_PRIMER_RNA_RECOGNITION_POSITIONS:
                    counts[3] += 1
        row_labels.append(f"{class_label(str(panel_row['design_class_id']))}  {short_candidate(candidate_id)}")
        matrix.append(counts)
    return region_labels, row_labels, matrix


def write_selected_substitutions_across_rt_plot(
    plot_root: Path,
    *,
    panel_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    mask_residues: list[dict[str, object]],
    input_hashes: dict[str, str | None],
    rt_annotation_context: RTAnnotationContext | None,
) -> dict[str, Any]:
    title = "Selected substitutions map to RT regions"
    ordered_panel = ordered_panel_rows(panel_rows)
    candidate_by_id = {str(row["candidate_id"]): row for row in candidate_rows if row.get("candidate_id")}
    ordered_residues = sorted(mask_residues, key=lambda row: int(row["canonical_position"]))
    positions = [int(row["canonical_position"]) for row in ordered_residues]
    residue_letters = [str(row.get("wt_aa") or "") for row in ordered_residues]
    position_index = {position: index for index, position in enumerate(positions)}
    missing_columns = {
        index
        for index, residue in enumerate(ordered_residues)
        if bool(residue.get("non_fixed_missing_backbone")) or not bool(residue.get("has_backbone_coordinates", True))
    }
    matrix: list[list[int]] = []
    row_labels: list[str] = []
    for panel_row in ordered_panel:
        candidate_id = str(panel_row["candidate_id"])
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            raise ValueError(f"Selection panel references missing candidate row: {candidate_id}")
        values = [7 if column in missing_columns else 0 for column in range(len(positions))]
        for mutation in canonical_mutations(candidate.get("canonical_mutations")):
            parsed = parse_mutation(mutation)
            position = int(parsed["position"])
            if position in position_index:
                values[position_index[position]] = mutation_category(str(parsed["wt"]), str(parsed["alt"]))
        matrix.append(values)
        row_labels.append(f"{class_label(str(panel_row['design_class_id']))}  {short_candidate(candidate_id)}")
    if not matrix:
        raise ValueError("selected-substitution plot requires selected candidates")
    fig, ax = plt.subplots(figsize=(12.2, max(3.8, 0.5 * len(matrix) + 2.0)))
    cmap = ListedColormap(
        [
            (247 / 255.0, 245 / 255.0, 239 / 255.0, 0.62),
            (102 / 255.0, 194 / 255.0, 165 / 255.0, 0.96),
            OKABE_ITO["blue"],
            OKABE_ITO["sky"],
            OKABE_ITO["orange"],
            OKABE_ITO["vermillion"],
            OKABE_ITO["purple"],
            "#d0d7de",
        ]
    )
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], cmap.N)
    ax.set_facecolor("#f8f7f2")
    if rt_annotation_context is not None:
        add_rt_annotation_context(ax, positions, row_count=len(matrix), context=rt_annotation_context)
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm, zorder=2)
    ax.set_yticks(list(range(len(row_labels))))
    ax.set_yticklabels(row_labels, fontsize=LABEL_SIZE - 0.5)
    tick_positions = position_tick_indices(len(positions))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(positions[index]) for index in tick_positions], fontsize=9.6)
    ax.set_xlabel("Residue position", fontsize=LABEL_SIZE, labelpad=8)
    top_axis = ax.secondary_xaxis("top")
    letter_tick_positions = list(range(len(positions)))
    top_axis.set_xticks(letter_tick_positions)
    top_axis.set_xticklabels([residue_letters[index] for index in letter_tick_positions], fontsize=2.8)
    for tick_label in top_axis.get_xticklabels():
        tick_label.set_fontfamily("DejaVu Sans Mono")
    top_axis.tick_params(length=0, pad=4)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=58)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [
        Patch(facecolor="#f7f5ef", edgecolor="#d8dee4", label="WT retained"),
        Patch(facecolor="#66c2a5", edgecolor="#ffffff", label="Changed"),
        Patch(facecolor=OKABE_ITO["blue"], edgecolor="#ffffff", label="Basic gained"),
        Patch(facecolor=OKABE_ITO["sky"], edgecolor="#ffffff", label="Basic lost"),
        Patch(facecolor=OKABE_ITO["orange"], edgecolor="#ffffff", label="Acidic gained"),
        Patch(facecolor=OKABE_ITO["vermillion"], edgecolor="#ffffff", label="Acidic lost"),
        Patch(facecolor=OKABE_ITO["purple"], edgecolor="#ffffff", label="Pro/Gly gained"),
        Patch(facecolor="#d0d7de", edgecolor="#ffffff", label="No backbone"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_SIZE,
        columnspacing=1.0,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.28, right=0.985, top=0.68, bottom=0.28)
    path = plot_root / "selection_selected_substitutions_across_rt.svg"
    alt = (
        "Heatmap of selected Eco1 RT candidates by residue position. Colored cells mark designed substitutions "
        "grouped by chemistry class; off-white cells retain WT amino acid identity. Display bands mark audited "
        "RT1-RT7 intervals and motif-anchor neighborhoods when annotation sources are available."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_selected_substitutions_across_rt",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Merges selected mutation positions and substitution chemistry into one sequence-position view with "
            "display-only RT region and motif context when the annotation ontology is available."
        ),
        interpretation_limit=(
            "Substitution chemistry is descriptive review context. It does not establish RT activity, processivity, "
            "strand displacement, or assay readiness."
        ),
        render_mode="wide_visual",
    )


def write_regional_mutation_burden_plot(
    plot_root: Path,
    *,
    panel_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    mask_residues: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    title = "Selected mutation burden by region"
    region_labels, row_labels, matrix = build_regional_mutation_burden_matrix(
        panel_rows=panel_rows,
        candidate_rows=candidate_rows,
        mask_residues=mask_residues,
    )
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    image = ax.imshow(matrix, aspect="equal", interpolation="nearest", cmap="YlOrBr")
    ax.set_yticks(list(range(len(row_labels))))
    ax.set_yticklabels(row_labels, fontsize=LABEL_SIZE - 0.5)
    ax.set_xticks(list(range(len(region_labels))))
    ax.set_xticklabels(region_labels, fontsize=LABEL_SIZE - 1, rotation=22, ha="right")
    max_count = max((max(values) for values in matrix), default=0)
    for row_index, values in enumerate(matrix):
        for col_index, value in enumerate(values):
            ax.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                fontsize=9.4,
                color=matrix_text_color(float(value), max_value=float(max_count)),
            )
    ax.set_title(title, fontsize=TITLE_SIZE, pad=12)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Mutation count", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=0.32, right=0.94, top=0.88, bottom=0.25)
    path = plot_root / "selection_regional_mutation_burden.svg"
    alt = (
        "Heatmap of selected Eco1 RT candidates by mutation count in catalytic/contact, near retained DNA/RNA, "
        "thumb-track, C-terminal primer-RNA recognition, and distal regions."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_regional_mutation_burden",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Summarizes selected substitutions by RT region so the panel can be read as regional review context "
            "rather than an activity ranking. The C-terminal column is an overlapping review context."
        ),
        interpretation_limit=(
            "Regional mutation burden is not a functional predictor. A zero thumb-contact-track count should not be "
            "read as direct thumb-track optimization."
        ),
        render_mode="wide_visual",
    )


def _regional_counts_from_panel_trace(panel_row: dict[str, object]) -> list[int] | None:
    try:
        trace = tie_break_trace(panel_row)
    except (KeyError, ValueError):
        return None
    required = {
        "catalytic_or_direct_contact_mutation_count",
        "nucleic_acid_facing_mutation_count",
        "thumb_contact_track_mutation_count",
        "c_terminal_primer_rna_recognition_mutation_count",
        "distal_scaffold_mutation_count",
    }
    if not required.issubset(trace):
        return None
    thumb_count = int(trace["thumb_contact_track_mutation_count"] or 0)
    near_dna_rna_count = max(int(trace["nucleic_acid_facing_mutation_count"] or 0) - thumb_count, 0)
    return [
        int(trace["catalytic_or_direct_contact_mutation_count"] or 0),
        near_dna_rna_count,
        thumb_count,
        int(trace["c_terminal_primer_rna_recognition_mutation_count"] or 0),
        int(trace["distal_scaffold_mutation_count"] or 0),
    ]


def _regional_bucket_index(position: int, residue: dict[str, object]) -> int:
    if _is_catalytic_or_direct_contact(residue):
        return 0
    if position in WANG_THUMB_CONTACT_TRACK_POSITIONS:
        return 2
    distance = _retained_na_distance(residue)
    if distance is not None and distance <= NA_FACING_DISTANCE_ANGSTROM:
        return 1
    return 4


def _is_catalytic_or_direct_contact(residue: dict[str, object]) -> bool:
    distance = _retained_na_distance(residue)
    return (
        bool(residue.get("motif_protected"))
        or bool(residue.get("wang_ec86_direct_contact_prior"))
        or bool(residue.get("direct_retained_dna_rna_contact_5a"))
        or (distance is not None and distance <= DIRECT_CONTACT_DISTANCE_ANGSTROM)
    )


def _retained_na_distance(residue: dict[str, object]) -> float | None:
    for field in ("nearest_context_atom_distance_angstrom", "distance_to_retained_na_angstrom"):
        value = residue.get(field)
        if value is not None:
            return float(value)
    return None
