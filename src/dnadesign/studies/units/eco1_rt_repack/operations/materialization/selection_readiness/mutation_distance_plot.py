"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/mutation_distance_plot.py

Mutation-set dissimilarity plot for Eco1 RT selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.mutation_distance import (
    canonical_mutation_positions,
    canonical_mutation_tokens,
    jaccard_distance,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.plot_support import (
    ordered_panel_rows,
    plot_row,
    short_candidate,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.visual_inventory import (
    SELECTION_PLOT_PLAIN_TITLES,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.shared.rendering import (
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


def write_selected_mutation_dissimilarity_plot(
    plot_root: Path,
    *,
    panel_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    """Write the selected-panel mutation-set dissimilarity audit plot."""

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


def _hamming_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right, strict=False)) + abs(len(left) - len(right))


__all__ = [
    "build_selected_mutation_dissimilarity_matrices",
    "build_selected_sequence_distance_matrix",
    "write_selected_mutation_dissimilarity_plot",
]
