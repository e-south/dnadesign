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
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.plot_support import (
    class_label,
    plot_row,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.shared.rendering import (
    LABEL_SIZE,
    LEGEND_SIZE,
    OKABE_ITO,
    TITLE_SIZE,
    save_accessible_svg,
    style_open_axes,
)

from ..shared.rt_annotation_context import RTAnnotationContext
from .chemistry_balance import write_na_facing_chemistry_balance_plot
from .local_structure_plot import (
    write_local_structure_by_region_plot,
    write_local_structure_stratification_plot,
    write_local_structure_threshold_sensitivity_plot,
)
from .mutation_distance_plot import (
    write_selected_mutation_dissimilarity_plot,
)
from .premise_alignment import write_premise_alignment_plot
from .region_msa_support_plot import write_regionwise_msa_support_plot
from .regional_plots import (
    write_regional_mutation_burden_plot,
    write_selected_substitutions_across_rt_plot,
)
from .sankey_plot import write_primary_panel_sankey_plot
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
    primary_panel_selection_trace_rows: list[dict[str, object]],
    input_hashes: dict[str, str | None],
    rt_annotation_context: RTAnnotationContext | None = None,
) -> list[dict[str, Any]]:
    """Write panel-selection plots and return manifest rows."""

    plot_root.mkdir(parents=True, exist_ok=True)
    _remove_retired_selection_plots(plot_root)
    return [
        _write_design_class_contrast(plot_root, triage_rows, panel_rows, input_hashes),
        write_primary_panel_sankey_plot(
            plot_root,
            primary_panel_selection_trace_rows=primary_panel_selection_trace_rows,
            input_hashes=input_hashes,
        ),
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
        write_selected_substitutions_across_rt_plot(
            plot_root,
            panel_rows=panel_rows,
            candidate_rows=candidate_rows,
            mask_residues=mask_residues,
            input_hashes=input_hashes,
            rt_annotation_context=rt_annotation_context,
        ),
        _write_design_class_gate_counts(plot_root, triage_rows, panel_rows, input_hashes),
        write_premise_alignment_plot(
            plot_root,
            panel_rows=panel_rows,
            triage_rows=triage_rows,
            input_hashes=input_hashes,
        ),
        write_selected_mutation_dissimilarity_plot(
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
    selected_counts = Counter(str(row["design_class_id"]) for row in panel_rows)
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
        selected_count = selected_counts[spec.design_class_id]
        if selected_count:
            ax.text(
                max(left[y_position], 1) + 1.25,
                y_position,
                f"selected {selected_count}",
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
        "and candidates missing gate inputs for each Eco1 design class. Selected-row counts are shown by class."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_design_class_gate_counts",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Counts candidates by protein-level gate outcome in each design class. Design class is review context, "
            "not a primary-panel quota."
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
    primary_counts = Counter(
        str(row["design_class_id"])
        for row in triage_rows
        if str(row.get("selection_candidate_tier") or "") == "primary_panel_candidate"
    )
    selected_by_class: dict[str, list[dict[str, object]]] = {}
    for row in panel_rows:
        selected_by_class.setdefault(str(row["design_class_id"]), []).append(row)
    rows: list[list[str]] = []
    for spec in ALL_SPECS:
        selected_rows = selected_by_class.get(spec.design_class_id, [])
        thumb_count = sum(int(row.get("thumb_contact_track_mutation_count") or 0) for row in selected_rows)
        c_terminal_count = sum(
            int(row.get("c_terminal_primer_rna_recognition_mutation_count") or 0) for row in selected_rows
        )
        near_count = sum(
            max(
                int(row.get("nucleic_acid_facing_mutation_count") or 0)
                - int(row.get("thumb_contact_track_mutation_count") or 0),
                0,
            )
            for row in selected_rows
        )
        rows.append(
            [
                class_label(spec.design_class_id),
                _profile_label(spec.conservation_profile_id),
                f">= {spec.conservation_threshold:.0%}",
                f"<= {spec.contact_threshold_angstrom:g} A",
                str(eligible_counts[spec.design_class_id]),
                str(primary_counts[spec.design_class_id]),
                str(len(selected_rows)),
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
        "Broad\npass",
        "Primary\ncandidates",
        "Selected",
        "Near retained\nDNA/RNA",
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
        colWidths=[0.15, 0.1, 0.075, 0.075, 0.075, 0.115, 0.08, 0.12, 0.11, 0.1],
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
        elif col_index in {7, 8, 9}:
            cell.set_facecolor("#eef6ff" if int(rows[row_index - 1][col_index]) else "#f7f5ef")
        else:
            cell.set_facecolor("#ffffff")
    ax.set_title(title, fontsize=TITLE_SIZE, pad=12)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.06)
    path = plot_root / "selection_design_class_contrast.svg"
    zero_thumb = all(int(row[7]) == 0 for row in rows)
    thumb_note = (
        " The selected primary panel does not mutate the declared Wang thumb-contact track." if zero_thumb else ""
    )
    alt = (
        "Table-like summary of Eco1 design classes showing MSA set, conservation threshold, retained DNA/RNA contact "
        "shell, broad-pass count, primary-candidate count, selected-row count, near retained DNA/RNA edits, "
        "thumb-track edits, and C-terminal primer-RNA recognition-region edits."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_design_class_contrast",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Shows how the declared mask policies differ across conservation denominator, conservation threshold, "
            "and retained-DNA/RNA contact shell. Broad-pass, primary-candidate, and selected-row counts are shown "
            "separately because design class is context, not a selection quota. Near retained DNA/RNA, thumb-track, "
            "and C-terminal edit counts are shown separately. The C-terminal count is an overlapping review context. "
            + thumb_note
        ),
        interpretation_limit=(
            "Design-class contrast explains review coverage. It does not establish function and does not require one "
            "selected row per mask policy."
        ),
        render_mode="wide_visual",
    )


def _profile_label(profile_id: str) -> str:
    if profile_id == "ec86_clade9_conservation_v1":
        return "Ec86 clade 9"
    if profile_id == "ec86_iia3_cluster42_1_conservation_v1":
        return "II-A3 42_1"
    return profile_id
