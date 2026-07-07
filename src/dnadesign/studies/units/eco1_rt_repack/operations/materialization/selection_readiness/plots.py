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
    class_label,
    matrix_text_color,
    ordered_panel_rows,
    plot_row,
    short_candidate,
)

from ..review_deliverables.rt_annotation_context import RTAnnotationContext
from .chemistry_balance import write_na_facing_chemistry_balance_plot
from .local_structure_plot import (
    write_local_structure_by_region_plot,
    write_local_structure_stratification_plot,
    write_local_structure_threshold_sensitivity_plot,
)
from .premise_alignment import write_premise_alignment_plot
from .region_msa_support_plot import write_regionwise_msa_support_plot
from .regional_plots import (
    write_regional_mutation_burden_plot,
    write_selected_substitutions_across_rt_plot,
)
from .visual_inventory import RETIRED_SELECTION_PLOT_FILE_NAMES, SELECTION_PLOT_PLAIN_TITLES

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_STATUS_ORDER = ("eligible", "ineligible", "missing_inputs")
_STATUS_COLORS = {
    "eligible": OKABE_ITO["green"],
    "ineligible": "#8c959f",
    "missing_inputs": OKABE_ITO["vermillion"],
}
_STATUS_LABELS = {
    "eligible": "Passes protein gate",
    "ineligible": "Blocked by gate",
    "missing_inputs": "Missing gate input",
}


def write_selection_readiness_plots(
    *,
    plot_root: Path,
    triage_rows: list[dict[str, object]],
    panel_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    mask_residues: list[dict[str, object]],
    local_structure_rows: list[dict[str, object]],
    local_structure_threshold_sensitivity_rows: list[dict[str, object]],
    region_msa_support_rows: list[dict[str, object]],
    input_hashes: dict[str, str | None],
    rt_annotation_context: RTAnnotationContext | None = None,
) -> list[dict[str, Any]]:
    """Write panel-selection plots and return manifest rows."""

    plot_root.mkdir(parents=True, exist_ok=True)
    _remove_retired_selection_plots(plot_root)
    return [
        _write_design_class_gate_counts(plot_root, triage_rows, panel_rows, input_hashes),
        _write_design_class_contrast(plot_root, triage_rows, panel_rows, input_hashes),
        write_local_structure_stratification_plot(
            plot_root,
            triage_rows=triage_rows,
            panel_rows=panel_rows,
            local_structure_rows=local_structure_rows,
            input_hashes=input_hashes,
        ),
        write_local_structure_threshold_sensitivity_plot(
            plot_root,
            threshold_sensitivity_rows=local_structure_threshold_sensitivity_rows,
            input_hashes=input_hashes,
        ),
        write_local_structure_by_region_plot(
            plot_root,
            panel_rows=panel_rows,
            local_structure_rows=local_structure_rows,
            input_hashes=input_hashes,
        ),
        _write_class_local_percentiles(plot_root, triage_rows, panel_rows, input_hashes),
        write_premise_alignment_plot(
            plot_root,
            panel_rows=panel_rows,
            triage_rows=triage_rows,
            input_hashes=input_hashes,
        ),
        write_selected_substitutions_across_rt_plot(
            plot_root,
            panel_rows=panel_rows,
            candidate_rows=candidate_rows,
            mask_residues=mask_residues,
            input_hashes=input_hashes,
            rt_annotation_context=rt_annotation_context,
        ),
        write_regional_mutation_burden_plot(
            plot_root,
            panel_rows=panel_rows,
            candidate_rows=candidate_rows,
            mask_residues=mask_residues,
            input_hashes=input_hashes,
        ),
        write_na_facing_chemistry_balance_plot(
            plot_root,
            panel_rows=panel_rows,
            triage_rows=triage_rows,
            input_hashes=input_hashes,
        ),
        write_regionwise_msa_support_plot(
            plot_root,
            panel_rows=panel_rows,
            region_msa_support_rows=region_msa_support_rows,
            input_hashes=input_hashes,
        ),
        _write_selected_sequence_distance(
            plot_root,
            panel_rows=panel_rows,
            candidate_rows=candidate_rows,
            input_hashes=input_hashes,
        ),
    ]


def _remove_retired_selection_plots(plot_root: Path) -> None:
    for file_name in RETIRED_SELECTION_PLOT_FILE_NAMES:
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


def _hamming_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right, strict=False)) + abs(len(left) - len(right))


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
    if direction == "moderate":
        sorted_values = sorted(values)
        midpoint = len(sorted_values) // 2
        if len(sorted_values) % 2:
            median = sorted_values[midpoint]
        else:
            median = (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0
        max_distance = max(abs(value - median) for value in sorted_values)
        if max_distance == 0.0:
            return 100.0
        return max(0.0, 100.0 * (1.0 - abs(selected_value - median) / max_distance))
    raise ValueError(f"Unknown class-local percentile direction: {direction}")


def _float_value(value: object, *, default: float = 0.0) -> float:
    return default if value is None else float(value)


def _float_or_none(value: object) -> float | None:
    return None if value is None else float(value)


def _write_design_class_gate_counts(
    plot_root: Path,
    triage_rows: list[dict[str, object]],
    panel_rows: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    title = SELECTION_PLOT_PLAIN_TITLES["selection_design_class_gate_counts"]
    counts_by_class: dict[str, Counter[str]] = {spec.design_class_id: Counter() for spec in ALL_SPECS}
    for row in triage_rows:
        class_id = str(row["design_class_id"])
        if class_id not in counts_by_class:
            raise ValueError(f"Unknown design class id in triage rows: {class_id}")
        counts_by_class[class_id][str(row["hard_gate_status"])] += 1
    selected_by_class = {str(row["design_class_id"]): str(row["candidate_id"]) for row in panel_rows}
    labels = [class_label(spec.design_class_id) for spec in ALL_SPECS]
    y_positions = list(range(len(ALL_SPECS)))
    fig, ax = plt.subplots(figsize=(7.8, 7.8))
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
    ax.set_box_aspect(1)
    style_open_axes(ax, grid=False)
    ax.grid(axis="x", color="#d0d7de", alpha=0.42, linewidth=0.7)
    ax.set_xlim(0, max(left) + 16)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_SIZE,
        columnspacing=1.2,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.27, right=0.97, top=0.87, bottom=0.28)
    path = plot_root / "selection_design_class_gate_counts.svg"
    alt = (
        "Stacked horizontal bars show candidates that pass the protein gate, candidates blocked by gate checks, "
        "and candidates missing gate inputs for each Eco1 design class. Each class has one selected candidate label."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_design_class_gate_counts",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Counts candidates by protein-level gate outcome in each design class before choosing one representative."
        ),
        interpretation_limit=(
            "Counts show panel preparation only. They do not measure RT activity, processivity, strand "
            "displacement, or structured-template readthrough."
        ),
        render_mode="wide_visual",
    )


def _write_design_class_contrast(
    plot_root: Path,
    triage_rows: list[dict[str, object]],
    panel_rows: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    title = SELECTION_PLOT_PLAIN_TITLES["selection_design_class_contrast"]
    eligible_counts = Counter(
        str(row["design_class_id"]) for row in triage_rows if str(row.get("hard_gate_status") or "") == "eligible"
    )
    triage_by_id = {str(row["candidate_id"]): row for row in triage_rows if row.get("candidate_id")}
    selected_by_class = {str(row["design_class_id"]): str(row["candidate_id"]) for row in panel_rows}
    rows: list[list[str]] = []
    for spec in ALL_SPECS:
        selected_id = selected_by_class.get(spec.design_class_id, "")
        selected_row = triage_by_id.get(selected_id, {})
        thumb_count = int(selected_row.get("thumb_contact_track_mutation_count") or 0)
        c_terminal_count = int(selected_row.get("c_terminal_primer_rna_recognition_mutation_count") or 0)
        near_count = max(int(selected_row.get("nucleic_acid_facing_mutation_count") or 0) - thumb_count, 0)
        rows.append(
            [
                class_label(spec.design_class_id),
                _profile_label(spec.conservation_profile_id),
                f">= {spec.conservation_threshold:.0%}",
                f"<= {spec.contact_threshold_angstrom:g} A",
                str(eligible_counts[spec.design_class_id]),
                short_candidate(selected_id) if selected_id else "missing",
                str(near_count),
                str(thumb_count),
                str(c_terminal_count),
            ]
        )
    columns = [
        "Class",
        "MSA",
        "WT %",
        "Contact",
        "Pass",
        "Selected",
        "Near DNA/RNA",
        "Thumb track",
        "C-term",
    ]
    fig, ax = plt.subplots(figsize=(10.4, 5.1))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.16, 0.11, 0.08, 0.08, 0.07, 0.14, 0.13, 0.12, 0.11],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.55)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        cell.set_linewidth(0.6)
        if row_index == 0:
            cell.set_facecolor("#f6f8fa")
            cell.set_text_props(weight="bold", color="#24292f")
        elif col_index in {6, 7, 8}:
            cell.set_facecolor("#eef6ff" if int(rows[row_index - 1][col_index]) else "#f7f5ef")
        else:
            cell.set_facecolor("#ffffff")
    ax.set_title(title, fontsize=TITLE_SIZE, pad=12)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.06)
    path = plot_root / "selection_design_class_contrast.svg"
    zero_thumb = all(int(row[7]) == 0 for row in rows)
    thumb_note = " The selected six do not mutate the declared Wang thumb-contact track." if zero_thumb else ""
    alt = (
        "Table-like summary of Eco1 design classes showing MSA set, conservation threshold, retained DNA/RNA contact "
        "shell, gate-pass count, selected row, near DNA/RNA edits, thumb-track edits, and C-terminal primer-RNA "
        "recognition-region edits."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_design_class_contrast",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Shows that the six panel slots are declared mask-policy contrasts across conservation denominator, "
            "conservation threshold, and retained-DNA/RNA contact shell. "
            "Near DNA/RNA, thumb-track, and C-terminal edit counts are shown separately. The C-terminal count is "
            "an overlapping review context. " + thumb_note
        ),
        interpretation_limit=(
            "Design-class contrast explains review coverage. It does not establish function and does not make the "
            "selected rows a global top-six ranking."
        ),
        render_mode="wide_visual",
    )


def _write_class_local_percentiles(
    plot_root: Path,
    triage_rows: list[dict[str, object]],
    panel_rows: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    title = SELECTION_PLOT_PLAIN_TITLES["selection_class_local_percentiles"]
    selected_by_class = {str(row["design_class_id"]): str(row["candidate_id"]) for row in panel_rows}
    triage_by_id = {str(row["candidate_id"]): row for row in triage_rows}
    metrics = [
        ("selection_support_alt_observed_fraction", "MSA support", "higher"),
        ("selection_support_unobserved_mutation_count", "Unsupported changes", "lower"),
        ("nucleic_acid_facing_chemistry_warning_count", "Chemistry warnings", "lower"),
        ("nucleic_acid_facing_mutation_count", "Regional burden", "moderate"),
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
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    image = ax.imshow(matrix, aspect="equal", interpolation="nearest", cmap="YlGnBu", vmin=0, vmax=100)
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
                color=matrix_text_color(value, max_value=max_percentile),
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("Favorable within-class percentile", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=0.3, right=0.93, top=0.88, bottom=0.24)
    path = plot_root / "selection_class_local_percentiles.svg"
    alt = (
        "Heatmap showing each selected candidate as a within-class percentile for MSA support, unsupported "
        "substitutions, near-DNA/RNA chemistry warnings, regional mutation burden, pLDDT, and WT RMSD. Higher "
        "percentiles are more favorable after direction handling."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_class_local_percentiles",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Compares each selected row only against candidates from the same design class. Higher percentiles mean "
            "more favorable within-class placement after direction handling: higher MSA support and pLDDT, fewer "
            "unsupported changes, fewer chemistry warnings, lower RMSD, and regional burden closer to the class "
            "median. The plot does not create a composite score."
        ),
        interpretation_limit=(
            "Percentiles explain panel review context. They are not activity, processivity, or strand-displacement "
            "measurements."
        ),
        render_mode="wide_visual",
    )


def _profile_label(profile_id: str) -> str:
    if profile_id == "ec86_clade9_conservation_v1":
        return "Ec86 clade 9"
    if profile_id == "ec86_iia3_cluster42_1_conservation_v1":
        return "II-A3 42_1"
    return profile_id


def _write_selected_sequence_distance(
    plot_root: Path,
    *,
    panel_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    title = SELECTION_PLOT_PLAIN_TITLES["selection_six_sequence_distance"]
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
                color=matrix_text_color(float(value), max_value=float(max_distance)),
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
