"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/sequence_distance_plot.py

Pairwise amino-acid difference plot for the selected Eco1 RT panel.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.plot_support import (
    matrix_text_color,
    ordered_panel_rows,
    plot_row,
    short_selected_variant,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.visual_inventory import (
    SELECTION_PLOT_PLAIN_TITLES,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.shared.policy_visuals import (
    policy_color,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.shared.rendering import (
    LABEL_SIZE,
    TICK_SIZE,
    TITLE_SIZE,
    save_accessible_svg,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def build_selected_sequence_distance_matrix(
    *,
    panel_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
) -> tuple[list[str], list[list[int]]]:
    """Return selected candidate ids and pairwise amino-acid Hamming counts."""

    ordered_panel, sequences = _selected_sequences(panel_rows=panel_rows, candidate_rows=candidate_rows)
    labels = [str(row["candidate_id"]) for row in ordered_panel]
    matrix = _distance_matrix(sequences)
    return labels, matrix


def write_selected_sequence_distance_plot(
    plot_root: Path,
    *,
    panel_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    """Write a lower-triangle heatmap of selected pairwise sequence differences."""

    title = SELECTION_PLOT_PLAIN_TITLES["selection_pairwise_sequence_differences"]
    ordered_panel, sequences = _selected_sequences(panel_rows=panel_rows, candidate_rows=candidate_rows)
    matrix = _distance_matrix(sequences)
    display_labels = [short_selected_variant(row) for row in ordered_panel]
    policy_ids = [str(row.get("policy_id") or "") for row in ordered_panel]
    matrix_values = np.asarray(matrix, dtype=float)
    pairwise_values = [
        int(matrix_values[row_index, col_index])
        for row_index in range(1, len(matrix))
        for col_index in range(row_index)
    ]
    if not pairwise_values:
        raise ValueError("Selected sequence-distance plot requires at least two selected sequences")
    minimum_distance = min(pairwise_values)
    maximum_distance = max(pairwise_values)
    upper_triangle_with_diagonal = np.triu(np.ones_like(matrix_values, dtype=bool), k=0)
    masked_matrix = np.ma.array(matrix_values, mask=upper_triangle_with_diagonal)
    colormap = plt.get_cmap("Blues").copy()
    colormap.set_bad("#F6F8FA")

    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    image = ax.imshow(
        masked_matrix,
        aspect="equal",
        interpolation="nearest",
        cmap=colormap,
        vmin=0,
        vmax=max(maximum_distance, 1),
    )
    indices = list(range(len(display_labels)))
    ax.set_xticks(indices)
    ax.set_yticks(indices)
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=TICK_SIZE)
    ax.set_yticklabels(display_labels, fontsize=TICK_SIZE)
    for tick_label, policy_id in zip(ax.get_xticklabels(), policy_ids, strict=True):
        tick_label.set_color(policy_color(policy_id))
        tick_label.set_fontweight("bold")
    for tick_label, policy_id in zip(ax.get_yticklabels(), policy_ids, strict=True):
        tick_label.set_color(policy_color(policy_id))
        tick_label.set_fontweight("bold")
    for row_index in range(1, len(matrix)):
        for col_index in range(row_index):
            value = int(matrix[row_index][col_index])
            ax.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                fontsize=11.5,
                fontweight="bold",
                color=matrix_text_color(value, max_value=maximum_distance),
            )
    boundaries = [index + 0.5 for index in range(len(policy_ids) - 1) if policy_ids[index] != policy_ids[index + 1]]
    for boundary in boundaries:
        ax.axvline(boundary, color="#57606A", linewidth=1.0, alpha=0.70)
        ax.axhline(boundary, color="#57606A", linewidth=1.0, alpha=0.70)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=38)
    ax.text(
        0.5,
        1.015,
        f"Each unique pair is shown once; selected pairs differ at {minimum_distance}-{maximum_distance} positions",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#57606A",
    )
    ax.tick_params(axis="both", length=0)
    ax.set_xticks(np.arange(-0.5, len(matrix), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix), 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.035)
    colorbar.set_label("Differing amino-acid positions", fontsize=LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=TICK_SIZE)
    fig.subplots_adjust(left=0.16, right=0.88, top=0.86, bottom=0.16)

    sequence_length = len(sequences[0])
    scope_note = _sequence_scope_note(sequence_length)
    path = plot_root / "selection_pairwise_sequence_differences.svg"
    alt = (
        f"Lower-triangle heatmap of amino-acid difference counts for {len(matrix)} selected Eco1 RT sequences. "
        f"Each unique pair appears once. Counts range from {minimum_distance} to {maximum_distance} positions. "
        f"{scope_note}"
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_pairwise_sequence_differences",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Counts positions whose final amino acids differ between each pair of selected sequences. " + scope_note
        ),
        interpretation_limit=(
            "Amino-acid difference counts are not the Jaccard selection score. Cross-group counts partly reflect "
            "different designable-position sets and do not establish mechanistic or functional independence."
        ),
        render_mode="standard_visual",
        data_sources=["candidate_pool.parquet", "selection/candidate_selection_panel.parquet"],
    )


def _selected_sequences(
    *,
    panel_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[str]]:
    ordered_panel = ordered_panel_rows(panel_rows)
    candidate_by_id = {str(row["candidate_id"]): row for row in candidate_rows if row.get("candidate_id")}
    sequences: list[str] = []
    for panel_row in ordered_panel:
        candidate_id = str(panel_row["candidate_id"])
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            raise ValueError(f"Selection panel references missing candidate row: {candidate_id}")
        sequence = str(candidate.get("sequence") or "").strip().upper()
        if not sequence:
            raise ValueError(f"Selected sequence-distance plot requires a non-empty protein sequence: {candidate_id}")
        sequences.append(sequence)
    sequence_lengths = {len(sequence) for sequence in sequences}
    if len(sequence_lengths) != 1:
        raise ValueError(
            "Selected sequence-distance plot requires equal-length protein sequences; "
            f"observed lengths: {sorted(sequence_lengths)}"
        )
    return ordered_panel, sequences


def _hamming_distance(left: str, right: str) -> int:
    return sum(left_aa != right_aa for left_aa, right_aa in zip(left, right, strict=True))


def _distance_matrix(sequences: list[str]) -> list[list[int]]:
    return [[_hamming_distance(left, right) for right in sequences] for left in sequences]


def _sequence_scope_note(sequence_length: int) -> str:
    if sequence_length == 309:
        return (
            "Counts use mapped Eco1 residues 3-311; the 11 restored WT terminal residues are identical, so the same "
            "counts apply to the full 320-aa handoff sequences."
        )
    return f"Counts use the shared {sequence_length}-residue candidate sequence span."


__all__ = ["build_selected_sequence_distance_matrix", "write_selected_sequence_distance_plot"]
