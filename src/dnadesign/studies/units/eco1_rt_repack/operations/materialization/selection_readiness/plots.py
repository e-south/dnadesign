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
from matplotlib.lines import Line2D
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
    legend_sizes,
    mutation_category,
    ordered_panel_rows,
    parse_mutation,
    plot_row,
    position_tick_indices,
    short_candidate,
    tie_break_trace,
)

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


def write_selection_readiness_plots(
    *,
    plot_root: Path,
    triage_rows: list[dict[str, object]],
    panel_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    mask_residues: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> list[dict[str, Any]]:
    """Write panel-selection plots and return manifest rows."""

    plot_root.mkdir(parents=True, exist_ok=True)
    return [
        _write_design_class_gate_counts(plot_root, triage_rows, panel_rows, input_hashes),
        write_population_stratification_plot(plot_root, triage_rows, panel_rows, input_hashes),
        _write_panel_review_axes(plot_root, panel_rows, input_hashes),
        _write_panel_sequence_differences(
            plot_root,
            panel_rows=panel_rows,
            candidate_rows=candidate_rows,
            mask_residues=mask_residues,
            input_hashes=input_hashes,
        ),
        _write_panel_mutation_geography_chemistry(
            plot_root,
            panel_rows=panel_rows,
            candidate_rows=candidate_rows,
            mask_residues=mask_residues,
            input_hashes=input_hashes,
        ),
    ]


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


def _write_panel_review_axes(
    plot_root: Path,
    panel_rows: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    title = "The six selected candidates balance MSA support with mutation geography"
    traces = [tie_break_trace(row) for row in panel_rows]
    by_class = {str(trace["design_class_id"]): trace for trace in traces}
    ordered = [by_class[spec.design_class_id] for spec in ALL_SPECS if spec.design_class_id in by_class]
    if not ordered:
        raise ValueError("selection panel plot requires at least one selected candidate")
    y_positions = list(range(len(ordered)))
    labels = [class_label(str(trace["design_class_id"])) for trace in ordered]
    x_values = [float(trace["selection_support_alt_observed_fraction"]) for trace in ordered]
    warning_counts = [int(trace["nucleic_acid_facing_chemistry_warning_count"]) for trace in ordered]
    na_counts = [int(trace["nucleic_acid_facing_mutation_count"]) for trace in ordered]
    sizes = [80 + (count * 4.2) for count in na_counts]
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    scatter = ax.scatter(
        x_values,
        y_positions,
        s=sizes,
        c=warning_counts,
        cmap="OrRd",
        edgecolors="#24292f",
        linewidths=0.55,
        zorder=3,
    )
    for x_value, y_position, row in zip(x_values, y_positions, ordered, strict=True):
        ax.text(
            min(x_value + 0.018, 1.02),
            y_position,
            short_candidate(str(row["candidate_id"])),
            va="center",
            ha="left",
            fontsize=10.2,
            color="#57606a",
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=LABEL_SIZE)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.08)
    ax.set_xlabel("Designed substitutions observed in the selected MSA denominator", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=12)
    style_open_axes(ax)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.82, pad=0.02)
    cbar.ax.tick_params(labelsize=10.5)
    cbar.set_label("Chemistry warning count", fontsize=11.5)
    size_handles = []
    for value in legend_sizes(na_counts):
        size_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#fdd0a2",
                markeredgecolor="#24292f",
                markersize=max(5.0, (80 + (value * 4.2)) ** 0.5),
                label=f"{value:g} NA-facing mutations",
            )
        )
    ax.legend(
        handles=size_handles,
        frameon=False,
        fontsize=10.5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=len(size_handles),
    )
    fig.subplots_adjust(left=0.3, right=0.9, top=0.89, bottom=0.24)
    path = plot_root / "selection_panel_review_axes.svg"
    alt = (
        "Scatter plot for the six selected Eco1 candidates. X position is the fraction of designed "
        "substitutions observed in the selected MSA denominator, marker area is nucleic-acid-facing "
        "mutation count, and color is local chemistry warning count."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_panel_review_axes",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Shows natural-sequence support, nucleic-acid-facing mutation geography, and simple "
            "local-chemistry warnings for the six selected variants."
        ),
        interpretation_limit=(
            "These axes explain panel choice. They are not calibrated functional predictors and do not "
            "claim improved strand displacement."
        ),
        render_mode="wide_visual",
    )


def _write_panel_sequence_differences(
    plot_root: Path,
    *,
    panel_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    mask_residues: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    title = "Selected Eco1 candidates vary only at designable protein positions"
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
        values = [2 if column in missing_columns else 0 for column in range(len(positions))]
        for mutation in canonical_mutations(candidate.get("canonical_mutations")):
            parsed = parse_mutation(mutation)
            if parsed["position"] not in position_index:
                continue
            values[position_index[int(parsed["position"])]] = 1
        matrix.append(values)
        row_labels.append(f"{class_label(str(panel_row['design_class_id']))}  {short_candidate(candidate_id)}")
    if not matrix:
        raise ValueError("selection panel sequence-difference plot requires at least one selected candidate")
    fig_width = 11.6
    fig_height = max(3.4, 0.46 * len(matrix) + 1.75)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap = ListedColormap(["#f7f5ef", OKABE_ITO["vermillion"], "#d0d7de"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_yticks(list(range(len(row_labels))))
    ax.set_yticklabels(row_labels, fontsize=LABEL_SIZE - 0.5)
    tick_positions = position_tick_indices(len(positions))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(positions[index]) for index in tick_positions], fontsize=9.3, rotation=0)
    ax.set_xlabel("Ec86 canonical residue position", fontsize=LABEL_SIZE, labelpad=8)
    top_axis = ax.secondary_xaxis("top")
    top_axis.set_xticks(tick_positions)
    top_axis.set_xticklabels([residue_letters[index] for index in tick_positions], fontsize=9.3)
    top_axis.tick_params(length=0, pad=4)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=12)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [
        Patch(facecolor="#f7f5ef", edgecolor="#d8dee4", label="WT residue retained"),
        Patch(facecolor=OKABE_ITO["vermillion"], edgecolor="#ffffff", label="Designed residue differs"),
        Patch(facecolor="#d0d7de", edgecolor="#ffffff", label="No fixed-backbone context"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
        fontsize=LEGEND_SIZE,
        columnspacing=1.3,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.28, right=0.985, top=0.82, bottom=0.22)
    path = plot_root / "selection_panel_sequence_differences.svg"
    alt = (
        "Heatmap of selected Eco1 panel candidates by canonical residue position. Off-white cells retain the "
        "WT residue, red cells mark designed amino-acid differences, and gray cells mark missing "
        "fixed-backbone context."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_panel_sequence_differences",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Shows where the class-balanced selected panel changes the Ec86 RT protein sequence. The plot keeps "
            "the panel decision connected to actual amino-acid differences rather than model-derived scores."
        ),
        interpretation_limit=(
            "Sequence differences show panel diversity and mask compliance context. They do not predict "
            "expression, activity, strand displacement, or hairpin readthrough."
        ),
        render_mode="wide_visual",
    )


def _write_panel_mutation_geography_chemistry(
    plot_root: Path,
    *,
    panel_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    mask_residues: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    title = "Selected Eco1 candidates change distal scaffold residues with limited local chemistry shifts"
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
    protected_columns = {index for index, residue in enumerate(ordered_residues) if bool(residue.get("protected"))}
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
            if position not in position_index:
                continue
            values[position_index[position]] = mutation_category(str(parsed["wt"]), str(parsed["alt"]))
        matrix.append(values)
        row_labels.append(f"{class_label(str(panel_row['design_class_id']))}  {short_candidate(candidate_id)}")
    if not matrix:
        raise ValueError("selection panel chemistry plot requires at least one selected candidate")

    fig_width = 12.4
    fig_height = max(3.8, 0.5 * len(matrix) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap = ListedColormap(
        [
            "#f7f5ef",
            "#66c2a5",
            OKABE_ITO["blue"],
            OKABE_ITO["sky"],
            OKABE_ITO["orange"],
            OKABE_ITO["vermillion"],
            OKABE_ITO["purple"],
            "#d0d7de",
        ]
    )
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], cmap.N)
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    for column in sorted(protected_columns):
        ax.axvline(column - 0.5, color="#24292f", linewidth=0.12, alpha=0.12)
    ax.set_yticks(list(range(len(row_labels))))
    ax.set_yticklabels(row_labels, fontsize=LABEL_SIZE)
    tick_positions = position_tick_indices(len(positions))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(positions[index]) for index in tick_positions], fontsize=9.6)
    ax.set_xlabel("Ec86 canonical residue position", fontsize=LABEL_SIZE, labelpad=8)
    top_axis = ax.secondary_xaxis("top")
    top_axis.set_xticks(tick_positions)
    top_axis.set_xticklabels([residue_letters[index] for index in tick_positions], fontsize=9.6)
    top_axis.tick_params(length=0, pad=4)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=12)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [
        Patch(facecolor="#f7f5ef", edgecolor="#d8dee4", label="WT retained"),
        Patch(facecolor="#66c2a5", edgecolor="#ffffff", label="Changed, no charge class"),
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
    fig.subplots_adjust(left=0.28, right=0.985, top=0.82, bottom=0.28)
    path = plot_root / "selection_panel_mutation_geography_chemistry.svg"
    alt = (
        "Heatmap of six selected Eco1 candidates by Ec86 residue position. Off-white marks WT-retained "
        "positions; colored cells mark designed substitutions grouped by charge or proline/glycine change."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_panel_mutation_geography_chemistry",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Shows where the selected variants change amino-acid chemistry across the Ec86 sequence. The "
            "plot keeps structural review tied to explicit residue substitutions rather than ESMC or SAE scores."
        ),
        interpretation_limit=(
            "Chemistry categories are descriptive and local. They do not measure stability, expression, "
            "RT activity, strand displacement, or structured-template readthrough."
        ),
        render_mode="wide_visual",
    )
