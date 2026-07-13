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
import numpy as np

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.mutation_distance import (
    canonical_mutation_positions,
    canonical_mutation_tokens,
    jaccard_distance,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.plot_support import (
    ordered_panel_rows,
    plot_row,
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
    triage_rows: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    """Write the selected-panel mutation-set dissimilarity audit plot."""

    title = SELECTION_PLOT_PLAIN_TITLES["selection_mutation_set_dissimilarity"]
    labels, position_matrix, _token_matrix = build_selected_mutation_dissimilarity_matrices(
        panel_rows=panel_rows,
        candidate_rows=candidate_rows,
    )
    display_labels = [_selected_label(row) for row in ordered_panel_rows(panel_rows)]
    fig, (ax, context_ax) = plt.subplots(
        1,
        2,
        figsize=(11.8, 6.1),
        gridspec_kw={"width_ratios": (1.35, 1.0), "wspace": 0.30},
    )
    image = ax.imshow(position_matrix, aspect="equal", interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(list(range(len(display_labels))))
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(list(range(len(display_labels))))
    ax.set_yticklabels(display_labels, fontsize=10)
    for row_index, values in enumerate(position_matrix):
        for col_index, value in enumerate(values):
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.3,
                color="#24292f" if value < 0.55 else "#ffffff",
            )
    ax.set_title(title, fontsize=TITLE_SIZE, pad=12)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, shrink=0.74, pad=0.03)
    distance_label = r"Position-set distance  $d_J = 1 - |A \cap B| / |A \cup B|$"
    cbar.set_label(distance_label, fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    all_contract_distances, selected_distances = _within_policy_position_distance_context(
        panel_rows=panel_rows,
        candidate_rows=candidate_rows,
        triage_rows=triage_rows,
    )
    bins = np.linspace(0.0, 1.0, 21)
    counts, _, _ = context_ax.hist(
        all_contract_distances,
        bins=bins,
        color="#AAB4BE",
        alpha=0.72,
        edgecolor="#FFFFFF",
        linewidth=0.5,
        label="All same-group candidate pairs",
    )
    rug_height = max(float(np.max(counts)) * 0.075, 1.0)
    context_ax.vlines(
        selected_distances,
        0.0,
        rug_height,
        color="#C00000",
        linewidth=2.0,
        alpha=0.90,
        label="Selected same-group pairs",
    )
    context_ax.set_xlim(0.0, 1.0)
    context_ax.set_xlabel(distance_label, fontsize=12.5)
    context_ax.set_ylabel("Number of candidate pairs", fontsize=12.5)
    context_ax.set_title("Selected pairs compared with generated candidates", fontsize=TITLE_SIZE - 1, pad=12)
    context_ax.spines[["top", "right"]].set_visible(False)
    context_ax.tick_params(labelsize=11)
    context_ax.grid(axis="y", color="#D0D7DE", alpha=0.45, linewidth=0.7)
    context_ax.legend(frameon=False, fontsize=10.3, loc="upper right")
    fig.subplots_adjust(left=0.10, right=0.985, top=0.88, bottom=0.20)
    path = plot_root / "selection_mutation_set_dissimilarity.svg"
    matrix_size = len(labels)
    alt = (
        f"{matrix_size}-by-{matrix_size} heatmap of pairwise mutation-set dissimilarity among selected Eco1 RT "
        "candidates beside a histogram of mutated-position distances for all structurally retained pairs generated "
        "with the same fixed and open residue rules. "
        "Each heatmap cell reports one Jaccard distance between mutated-position sets. Red ticks mark pairs "
        "represented in the selected panel against pairs drawn from the same generation policy."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_mutation_set_dissimilarity",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Shows how much selected sequences reuse the same mutated positions and places those distances against "
            "all structurally retained candidate pairs generated under the same fixed and open residue rules. The "
            "first pair in each design group is chosen "
            "by exhaustive comparison; a third row maximizes its minimum distance from that pair. Exact-substitution "
            "distance remains a later selection tie-break and is reported in the panel table."
        ),
        interpretation_limit=(
            "Mutation-set dissimilarity reduces overlap but does not guarantee independent mechanisms or provide "
            "functional evidence."
        ),
        render_mode="compact_wide_visual",
    )


def _within_policy_position_distance_context(
    *,
    panel_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    triage_rows: list[dict[str, object]],
) -> tuple[list[float], list[float]]:
    candidate_by_id = {str(row["candidate_id"]): row for row in candidate_rows if row.get("candidate_id")}
    contract_ids = {
        str(row["candidate_id"])
        for row in triage_rows
        if row.get("candidate_id") and bool(row.get("selection_contract_pass"))
    }
    contract_by_policy: dict[str, list[frozenset[int]]] = {}
    for candidate_id in contract_ids:
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            continue
        policy_id = str(candidate.get("policy_id") or "")
        contract_by_policy.setdefault(policy_id, []).append(
            canonical_mutation_positions(candidate.get("canonical_mutations"))
        )
    selected_by_policy: dict[str, list[frozenset[int]]] = {}
    for panel_row in ordered_panel_rows(panel_rows):
        candidate_id = str(panel_row["candidate_id"])
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            raise ValueError(f"Selection panel references missing candidate row: {candidate_id}")
        policy_id = str(panel_row.get("policy_id") or candidate.get("policy_id") or "")
        selected_by_policy.setdefault(policy_id, []).append(
            canonical_mutation_positions(candidate.get("canonical_mutations"))
        )
    all_contract = [
        jaccard_distance(rows[left_index], rows[right_index])
        for rows in contract_by_policy.values()
        for left_index in range(len(rows))
        for right_index in range(left_index + 1, len(rows))
    ]
    selected = [
        jaccard_distance(rows[left_index], rows[right_index])
        for rows in selected_by_policy.values()
        for left_index in range(len(rows))
        for right_index in range(left_index + 1, len(rows))
    ]
    if not all_contract:
        raise ValueError("Mutation-distance context requires at least one contract-pass within-policy pair")
    if not selected:
        raise ValueError("Mutation-distance context requires at least one selected within-policy pair")
    return all_contract, selected


def _hamming_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right, strict=False)) + abs(len(left) - len(right))


def _selected_label(row: dict[str, object]) -> str:
    policy_id = str(row.get("policy_id") or "")
    prefix = {
        "distal_scaffold_repack_v1": "D",
        "near_dna_rna_acid_free_v1": "P",
        "combined_near_acid_free_plus_distal_v1": "C",
    }.get(policy_id, "V")
    rank = int(row.get("within_group_rank") or row.get("selection_rank") or 0)
    return f"{prefix}{rank}"


__all__ = [
    "build_selected_mutation_dissimilarity_matrices",
    "build_selected_sequence_distance_matrix",
    "write_selected_mutation_dissimilarity_plot",
]
