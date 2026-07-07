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
from .mutation_distance import (
    canonical_mutation_positions,
    canonical_mutation_tokens,
    jaccard_distance,
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
from matplotlib.patches import FancyBboxPatch, PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

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
        _write_primary_panel_sankey_plot(
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


def build_selected_mutation_dissimilarity_matrices(
    *,
    panel_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
) -> tuple[list[str], list[list[float]], list[list[float]]]:
    """Return pairwise mutated-position and exact-substitution Jaccard distances."""

    ordered_panel = ordered_panel_rows(panel_rows)
    candidate_by_id = {str(row["candidate_id"]): row for row in candidate_rows if row.get("candidate_id")}
    labels: list[str] = []
    position_sets: list[frozenset[int]] = []
    token_sets: list[frozenset[str]] = []
    for panel_row in ordered_panel:
        candidate_id = str(panel_row["candidate_id"])
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            raise ValueError(f"Selection panel references missing candidate row: {candidate_id}")
        labels.append(candidate_id)
        position_sets.append(canonical_mutation_positions(candidate.get("canonical_mutations")))
        token_sets.append(canonical_mutation_tokens(candidate.get("canonical_mutations")))
    position_matrix = [[round(jaccard_distance(left, right), 3) for right in position_sets] for left in position_sets]
    token_matrix = [[round(jaccard_distance(left, right), 3) for right in token_sets] for left in token_sets]
    return labels, position_matrix, token_matrix


def _hamming_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right, strict=False)) + abs(len(left) - len(right))


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


def _write_primary_panel_sankey_plot(
    plot_root: Path,
    *,
    primary_panel_selection_trace_rows: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    title = SELECTION_PLOT_PLAIN_TITLES["selection_primary_panel_sankey"]
    if not primary_panel_selection_trace_rows:
        raise ValueError("primary-panel Sankey plot requires selection trace rows")
    by_stage = {str(row["stage_id"]): row for row in primary_panel_selection_trace_rows}
    required = {
        "candidate_pool",
        "broad_contract_pool",
        "primary_panel_candidate_pool",
        "global_conservative_diverse_selection",
    }
    missing = required - set(by_stage)
    if missing:
        raise ValueError(f"Primary-panel Sankey plot is missing trace stages: {', '.join(sorted(missing))}")
    counts = {stage_id: int(by_stage[stage_id]["remaining_count"]) for stage_id in required}
    other_primary = max(counts["primary_panel_candidate_pool"] - counts["global_conservative_diverse_selection"], 0)
    max_flow = max(counts.values())
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    nodes = {
        "candidate_pool": (0.04, 0.47, "Accepted\ncandidates", counts["candidate_pool"], OKABE_ITO["blue"]),
        "broad_contract_pool": (
            0.28,
            0.47,
            "Broad protein\ncontract",
            counts["broad_contract_pool"],
            OKABE_ITO["green"],
        ),
        "primary_panel_candidate_pool": (
            0.53,
            0.52,
            "Primary panel\ncandidates",
            counts["primary_panel_candidate_pool"],
            OKABE_ITO["sky"],
        ),
        "global_conservative_diverse_selection": (
            0.79,
            0.66,
            "Selected primary\npanel",
            counts["global_conservative_diverse_selection"],
            OKABE_ITO["orange"],
        ),
        "other_primary": (
            0.79,
            0.41,
            "Other primary\ncandidates",
            other_primary,
            "#c9d1d9",
        ),
    }
    _draw_flow(
        ax,
        start=(0.22, 0.55),
        end=(0.28, 0.55),
        count=counts["broad_contract_pool"],
        max_count=max_flow,
        color=OKABE_ITO["green"],
        label=f"{counts['broad_contract_pool']} broad-contract rows",
    )
    _draw_flow(
        ax,
        start=(0.46, 0.62),
        end=(0.53, 0.6),
        count=counts["primary_panel_candidate_pool"],
        max_count=max_flow,
        color=OKABE_ITO["sky"],
        label=f"{counts['primary_panel_candidate_pool']} primary candidates",
    )
    _draw_flow(
        ax,
        start=(0.71, 0.74),
        end=(0.79, 0.74),
        count=counts["global_conservative_diverse_selection"],
        max_count=max_flow,
        color=OKABE_ITO["orange"],
        label=f"{counts['global_conservative_diverse_selection']} selected",
    )
    _draw_flow(
        ax,
        start=(0.71, 0.66),
        end=(0.79, 0.49),
        count=other_primary,
        max_count=max_flow,
        color="#c9d1d9",
        label=f"{other_primary} not selected",
    )
    for x, y, label, count, color in nodes.values():
        _draw_sankey_node(ax, x=x, y=y, label=label, count=count, color=color)
    ax.text(
        0.04,
        0.89,
        (
            f"{counts['candidate_pool']} accepted -> {counts['broad_contract_pool']} broad-contract rows -> "
            f"{counts['primary_panel_candidate_pool']} primary candidates -> "
            f"{counts['global_conservative_diverse_selection']} selected"
        ),
        ha="left",
        va="center",
        fontsize=10.8,
        color="#57606a",
    )
    ax.text(
        0.04,
        0.08,
        "The final step is a global conservative-diverse selection, not a design-class quota.",
        ha="left",
        va="center",
        fontsize=10.5,
        color="#57606a",
    )
    ax.set_title(title, fontsize=TITLE_SIZE, pad=10)
    path = plot_root / "selection_primary_panel_sankey.svg"
    alt = (
        "Sankey-style flow showing accepted Eco1 RT candidates, rows passing the preservation contract, rows in the "
        "primary candidate pool, and the selected conservative-diverse primary-panel rows."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_primary_panel_sankey",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Shows how the selector moves from accepted candidates through the preservation contract to a primary "
            "candidate pool and then to the final conservative-diverse selected panel."
        ),
        interpretation_limit=(
            "The flow is a protein-level selection record. It does not measure RT activity, processivity, or strand "
            "displacement."
        ),
        render_mode="wide_visual",
    )


def _draw_flow(
    ax: plt.Axes,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    count: int,
    max_count: int,
    color: str,
    label: str,
) -> None:
    _ = label
    if count <= 0:
        return
    width = 4.0 + 30.0 * (count / max(max_count, 1)) ** 0.5
    control_dx = max((end[0] - start[0]) * 0.55, 0.02)
    path = MplPath(
        [
            start,
            (start[0] + control_dx, start[1]),
            (end[0] - control_dx, end[1]),
            end,
        ],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            lw=width,
            alpha=0.34,
            capstyle="round",
            zorder=1,
        )
    )


def _draw_sankey_node(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    label: str,
    count: int,
    color: str,
) -> None:
    width = 0.18
    height = 0.14
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.016",
            linewidth=0.8,
            edgecolor="#d0d7de",
            facecolor="#ffffff",
            zorder=4,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            0.018,
            height,
            boxstyle="round,pad=0.0,rounding_size=0.012",
            linewidth=0,
            facecolor=color,
            alpha=0.95,
            zorder=5,
        )
    )
    ax.text(x + 0.03, y + 0.088, label, ha="left", va="center", fontsize=10.2, color="#24292f", zorder=6)
    ax.text(x + 0.03, y + 0.038, str(count), ha="left", va="center", fontsize=13.0, weight="bold", zorder=6)


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
    labels, position_matrix, token_matrix = build_selected_mutation_dissimilarity_matrices(
        panel_rows=panel_rows,
        candidate_rows=candidate_rows,
    )
    display_labels = [short_candidate(label) for label in labels]
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    image = ax.imshow(position_matrix, aspect="equal", interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(list(range(len(display_labels))))
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(list(range(len(display_labels))))
    ax.set_yticklabels(display_labels, fontsize=10)
    for row_index, values in enumerate(position_matrix):
        for col_index, value in enumerate(values):
            exact_value = token_matrix[row_index][col_index]
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}\n{exact_value:.2f}",
                ha="center",
                va="center",
                fontsize=8.3,
                color="#24292f" if value < 0.55 else "#ffffff",
            )
    ax.set_title(title, fontsize=TITLE_SIZE, pad=12)
    fig.text(
        0.5,
        0.045,
        "Cell text: mutation-position distance / exact-substitution distance",
        ha="center",
        va="center",
        fontsize=9.6,
        color="#57606a",
    )
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, shrink=0.74, pad=0.03)
    cbar.set_label("Mutated-position Jaccard distance", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=0.2, right=0.92, top=0.88, bottom=0.25)
    path = plot_root / "selection_six_sequence_distance.svg"
    alt = (
        "Six-by-six heatmap of pairwise mutation-set dissimilarity among selected Eco1 RT candidates. Cell text shows "
        "mutated-position Jaccard distance on the first line and exact-substitution Jaccard distance on the second."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_six_sequence_distance",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Audits whether selected rows reuse the same mutation positions or exact substitutions. Mutation-set "
            "dissimilarity is part of the global selection order after conservative safety fields."
        ),
        interpretation_limit="Mutation-set dissimilarity guards redundancy; it is not functional evidence.",
        render_mode="compact_wide_visual",
    )
