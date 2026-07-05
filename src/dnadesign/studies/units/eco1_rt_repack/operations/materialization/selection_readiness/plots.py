"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/plots.py

Panel-selection SVG plots for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import (
    ALL_SPECS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.rendering import (
    LABEL_SIZE,
    LEGEND_SIZE,
    OKABE_ITO,
    TITLE_SIZE,
    save_accessible_svg,
    style_open_axes,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.plot_support import (
    canonical_mutations,
    class_label,
    mutation_category,
    ordered_panel_rows,
    parse_mutation,
    plot_row,
    position_tick_indices,
    short_candidate,
    tie_break_trace,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.review_axes import (
    DIRECT_CONTACT_DISTANCE_ANGSTROM,
    NA_FACING_DISTANCE_ANGSTROM,
    WANG_THUMB_CONTACT_TRACK_POSITIONS,
)

from ..review_deliverables.design_class_mask_annotations import add_rt_annotation_context
from ..review_deliverables.rt_annotation_context import RTAnnotationContext
from .population_stratification import write_population_stratification_plot

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
_RETIRED_SELECTION_PLOT_FILES = (
    "selection_panel_review_axes.svg",
    "selection_panel_sequence_differences.svg",
    "selection_panel_mutation_geography_chemistry.svg",
)


def write_selection_readiness_plots(
    *,
    plot_root: Path,
    triage_rows: list[dict[str, object]],
    panel_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    mask_residues: list[dict[str, object]],
    input_hashes: dict[str, str | None],
    rt_annotation_context: RTAnnotationContext | None = None,
) -> list[dict[str, Any]]:
    """Write panel-selection plots and return manifest rows."""

    plot_root.mkdir(parents=True, exist_ok=True)
    _remove_retired_selection_plots(plot_root)
    return [
        _write_design_class_gate_counts(plot_root, triage_rows, panel_rows, input_hashes),
        write_population_stratification_plot(plot_root, triage_rows, panel_rows, input_hashes),
        _write_class_local_percentiles(plot_root, triage_rows, panel_rows, input_hashes),
        _write_selected_sequence_distance(
            plot_root,
            panel_rows=panel_rows,
            candidate_rows=candidate_rows,
            input_hashes=input_hashes,
        ),
        _write_selected_substitutions_across_rt(
            plot_root,
            panel_rows=panel_rows,
            candidate_rows=candidate_rows,
            mask_residues=mask_residues,
            input_hashes=input_hashes,
            rt_annotation_context=rt_annotation_context,
        ),
        _write_regional_mutation_burden(
            plot_root,
            panel_rows=panel_rows,
            candidate_rows=candidate_rows,
            mask_residues=mask_residues,
            input_hashes=input_hashes,
        ),
    ]


def _remove_retired_selection_plots(plot_root: Path) -> None:
    for file_name in _RETIRED_SELECTION_PLOT_FILES:
        path = plot_root / file_name
        if path.exists():
            path.unlink()


def build_selected_sequence_distance_matrix(
    *,
    panel_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
) -> tuple[list[str], list[list[int]]]:
    """Return selected candidate ids and pairwise protein-sequence distances."""

    ordered_panel = ordered_panel_rows(panel_rows)
    candidate_by_id = {str(row["candidate_id"]): row for row in candidate_rows if row.get("candidate_id")}
    labels: list[str] = []
    sequences: list[str] = []
    for panel_row in ordered_panel:
        candidate_id = str(panel_row["candidate_id"])
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            raise ValueError(f"Selection panel references missing candidate row: {candidate_id}")
        labels.append(candidate_id)
        sequences.append(str(candidate.get("sequence") or ""))
    matrix = [[_hamming_distance(left, right) for right in sequences] for left in sequences]
    return labels, matrix


def build_regional_mutation_burden_matrix(
    *,
    panel_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    mask_residues: list[dict[str, object]],
) -> tuple[list[str], list[str], list[list[int]]]:
    """Return selected-candidate mutation counts by RT review region."""

    region_labels = [
        "Catalytic or direct contact",
        "Near retained DNA/RNA annulus",
        "Thumb-contact track",
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
            counts = [0, 0, 0, 0]
            for mutation in canonical_mutations(candidate.get("canonical_mutations")):
                parsed = parse_mutation(mutation)
                position = int(parsed["position"])
                counts[_regional_bucket_index(position, residue_by_position.get(position, {}))] += 1
        row_labels.append(f"{class_label(str(panel_row['design_class_id']))}  {short_candidate(candidate_id)}")
        matrix.append(counts)
    return region_labels, row_labels, matrix


def _regional_counts_from_panel_trace(panel_row: dict[str, object]) -> list[int] | None:
    try:
        trace = tie_break_trace(panel_row)
    except (KeyError, ValueError):
        return None
    required = {
        "catalytic_or_direct_contact_mutation_count",
        "nucleic_acid_facing_mutation_count",
        "thumb_contact_track_mutation_count",
        "distal_scaffold_mutation_count",
    }
    if not required.issubset(trace):
        return None
    thumb_count = int(trace["thumb_contact_track_mutation_count"] or 0)
    near_annulus_count = max(int(trace["nucleic_acid_facing_mutation_count"] or 0) - thumb_count, 0)
    return [
        int(trace["catalytic_or_direct_contact_mutation_count"] or 0),
        near_annulus_count,
        thumb_count,
        int(trace["distal_scaffold_mutation_count"] or 0),
    ]


def _hamming_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right, strict=False)) + abs(len(left) - len(right))


def _regional_bucket_index(position: int, residue: dict[str, object]) -> int:
    if _is_catalytic_or_direct_contact(residue):
        return 0
    if position in WANG_THUMB_CONTACT_TRACK_POSITIONS:
        return 2
    distance = _retained_na_distance(residue)
    if distance is not None and distance <= NA_FACING_DISTANCE_ANGSTROM:
        return 1
    return 3


def _is_catalytic_or_direct_contact(residue: dict[str, object]) -> bool:
    distance = _retained_na_distance(residue)
    return (
        bool(residue.get("protected"))
        or bool(residue.get("motif_protected"))
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


def _class_percentile(
    *,
    selected_value: float,
    class_values: list[float],
    direction: str,
) -> float:
    values = [value for value in class_values if value == value]
    if not values:
        return 0.0
    if direction == "lower":
        return 100.0 * sum(value >= selected_value for value in values) / len(values)
    if direction == "higher":
        return 100.0 * sum(value <= selected_value for value in values) / len(values)
    raise ValueError(f"Unknown class-local percentile direction: {direction}")


def _float_value(value: object, *, default: float = 0.0) -> float:
    return default if value is None else float(value)


def _float_or_none(value: object) -> float | None:
    return None if value is None else float(value)


def _matrix_text_color(value: float, *, max_value: float) -> str:
    return "#ffffff" if max_value > 0 and value >= max_value * 0.55 else "#24292f"


def _write_design_class_gate_counts(
    plot_root: Path,
    triage_rows: list[dict[str, object]],
    panel_rows: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    title = "Each Eco1 design class retains fold-preserved candidates for panel selection"
    counts_by_class: dict[str, Counter[str]] = {spec.design_class_id: Counter() for spec in ALL_SPECS}
    for row in triage_rows:
        class_id = str(row["design_class_id"])
        if class_id not in counts_by_class:
            raise ValueError(f"Unknown design class id in triage rows: {class_id}")
        counts_by_class[class_id][str(row["hard_gate_status"])] += 1
    selected_by_class = {str(row["design_class_id"]): str(row["candidate_id"]) for row in panel_rows}
    labels = [class_label(spec.design_class_id) for spec in ALL_SPECS]
    y_positions = list(range(len(ALL_SPECS)))
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    left = [0] * len(ALL_SPECS)
    for status in _STATUS_ORDER:
        widths = [counts_by_class[spec.design_class_id][status] for spec in ALL_SPECS]
        ax.barh(
            y_positions,
            widths,
            left=left,
            height=0.62,
            color=_STATUS_COLORS[status],
            edgecolor="#ffffff",
            linewidth=0.65,
            label=_STATUS_LABELS[status],
        )
        left = [start + width for start, width in zip(left, widths, strict=True)]
    for y_position, spec in zip(y_positions, ALL_SPECS, strict=True):
        selected = selected_by_class.get(spec.design_class_id)
        if selected:
            ax.text(
                max(left[y_position], 1) + 1.25,
                y_position,
                short_candidate(selected),
                va="center",
                ha="left",
                fontsize=10.5,
                color="#57606a",
            )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=LABEL_SIZE)
    ax.invert_yaxis()
    ax.set_xlabel("Candidate count", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=12)
    style_open_axes(ax, grid=False)
    ax.grid(axis="x", color="#d0d7de", alpha=0.42, linewidth=0.7)
    ax.set_xlim(0, max(left) + 16)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=len(_STATUS_ORDER),
        frameon=False,
        fontsize=LEGEND_SIZE,
    )
    fig.subplots_adjust(left=0.27, right=0.98, top=0.9, bottom=0.22)
    path = plot_root / "selection_design_class_gate_counts.svg"
    alt = (
        "Stacked horizontal bars show eligible, manual-reserve, excluded, and missing-input candidates "
        "for each Eco1 design class. Each class has one selected candidate label."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_design_class_gate_counts",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Counts candidates that pass feasibility and fold checks in each design class before choosing "
            "one representative."
        ),
        interpretation_limit=(
            "Counts show panel preparation only. They do not measure RT activity, processivity, strand "
            "displacement, or structured-template readthrough."
        ),
        render_mode="wide_visual",
    )


def _write_class_local_percentiles(
    plot_root: Path,
    triage_rows: list[dict[str, object]],
    panel_rows: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    title = "Each selected row is reviewed within its own mask class"
    selected_by_class = {str(row["design_class_id"]): str(row["candidate_id"]) for row in panel_rows}
    triage_by_id = {str(row["candidate_id"]): row for row in triage_rows}
    metrics = [
        ("selection_support_alt_observed_fraction", "MSA support", "higher"),
        ("selection_support_unobserved_mutation_count", "Unsupported changes", "lower"),
        ("nucleic_acid_facing_mutation_count", "Near DNA/RNA or thumb", "higher"),
        ("nucleic_acid_facing_chemistry_warning_count", "Chemistry warnings", "lower"),
        ("mean_plddt", "pLDDT", "higher"),
        ("wt_runtime_ca_rmsd", "WT RMSD", "lower"),
    ]
    matrix: list[list[float]] = []
    row_labels: list[str] = []
    for spec in ALL_SPECS:
        candidate_id = selected_by_class.get(spec.design_class_id)
        if candidate_id is None:
            continue
        selected = triage_by_id.get(candidate_id)
        if selected is None:
            raise ValueError(f"Selected candidate is missing from triage rows: {candidate_id}")
        class_rows = [row for row in triage_rows if str(row.get("design_class_id") or "") == spec.design_class_id]
        matrix.append(
            [
                _class_percentile(
                    selected_value=_float_value(selected.get(metric)),
                    class_values=[_float_value(row.get(metric)) for row in class_rows if row.get(metric) is not None],
                    direction=direction,
                )
                for metric, _label, direction in metrics
            ]
        )
        row_labels.append(f"{class_label(spec.design_class_id)}  {short_candidate(candidate_id)}")
    if not matrix:
        raise ValueError("class-local percentile plot requires selected rows")
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_yticks(list(range(len(row_labels))))
    ax.set_yticklabels(row_labels, fontsize=LABEL_SIZE - 0.5)
    ax.set_xticks(list(range(len(metrics))))
    ax.set_xticklabels(
        [label for _metric, label, _direction in metrics], fontsize=LABEL_SIZE - 1, rotation=25, ha="right"
    )
    ax.set_title(title, fontsize=TITLE_SIZE, pad=12)
    ax.tick_params(axis="both", length=0)
    max_percentile = 100.0
    for row_index, values in enumerate(matrix):
        for col_index, value in enumerate(values):
            ax.text(
                col_index,
                row_index,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=9.2,
                color=_matrix_text_color(value, max_value=max_percentile),
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("Within-class percentile", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=0.3, right=0.93, top=0.88, bottom=0.24)
    path = plot_root / "selection_class_local_percentiles.svg"
    alt = (
        "Heatmap showing each selected candidate as a within-class percentile for MSA support, unsupported "
        "substitutions, near-DNA/RNA or thumb-track mutation count, chemistry warnings, pLDDT, and WT RMSD."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_class_local_percentiles",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Compares each selected row only against candidates from the same mask class. Percentiles summarize "
            "the lexicographic review variables without creating a composite score."
        ),
        interpretation_limit=(
            "Percentiles explain panel review context. They are not activity, processivity, or strand-displacement "
            "measurements."
        ),
        render_mode="wide_visual",
    )


def _write_selected_sequence_distance(
    plot_root: Path,
    *,
    panel_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    title = "The selected six sample distinct sequence neighborhoods"
    labels, matrix = build_selected_sequence_distance_matrix(panel_rows=panel_rows, candidate_rows=candidate_rows)
    display_labels = [short_candidate(label) for label in labels]
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    image = ax.imshow(matrix, aspect="equal", interpolation="nearest", cmap="Blues")
    ax.set_xticks(list(range(len(display_labels))))
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(list(range(len(display_labels))))
    ax.set_yticklabels(display_labels, fontsize=10)
    max_distance = max((max(values) for values in matrix), default=0)
    for row_index, values in enumerate(matrix):
        for col_index, value in enumerate(values):
            ax.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                fontsize=9.2,
                color=_matrix_text_color(float(value), max_value=float(max_distance)),
            )
    ax.set_title(title, fontsize=TITLE_SIZE, pad=12)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, shrink=0.74, pad=0.03)
    cbar.set_label("Pairwise amino-acid differences", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=0.2, right=0.92, top=0.88, bottom=0.2)
    path = plot_root / "selection_six_sequence_distance.svg"
    alt = "Six-by-six heatmap of pairwise amino-acid Hamming distances among the selected Eco1 RT candidates."
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_six_sequence_distance",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description="Shows whether the selected panel is sequence-redundant or spans different sequence neighborhoods.",
        interpretation_limit="Sequence distance is a diversity context metric, not functional evidence.",
        render_mode="compact_wide_visual",
    )


def _write_selected_substitutions_across_rt(
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
    top_axis.set_xticks(tick_positions)
    top_axis.set_xticklabels([residue_letters[index] for index in tick_positions], fontsize=9.4)
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


def _write_regional_mutation_burden(
    plot_root: Path,
    *,
    panel_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    mask_residues: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    title = "Selected candidates differ in which RT regions carry mutations"
    region_labels, row_labels, matrix = build_regional_mutation_burden_matrix(
        panel_rows=panel_rows,
        candidate_rows=candidate_rows,
        mask_residues=mask_residues,
    )
    fig, ax = plt.subplots(figsize=(9.6, max(3.8, 0.55 * len(matrix) + 1.8)))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="YlOrBr")
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
                color=_matrix_text_color(float(value), max_value=float(max_count)),
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
        "Heatmap of selected Eco1 RT candidates by mutation count in catalytic/contact, substrate-proximal, "
        "thumb-track, and distal regions."
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
            "rather than a global top-six ranking."
        ),
        interpretation_limit=(
            "Regional mutation burden is not a functional predictor. A zero thumb-contact-track count should not be "
            "read as direct thumb-track optimization."
        ),
        render_mode="wide_visual",
    )
