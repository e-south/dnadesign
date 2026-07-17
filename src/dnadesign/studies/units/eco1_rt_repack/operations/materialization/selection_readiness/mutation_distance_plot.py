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
    short_selected_variant,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.visual_inventory import (
    SELECTION_PLOT_PLAIN_TITLES,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.shared.policy_visuals import (
    POLICY_ORDER,
    policy_color,
    policy_label,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.shared.rendering import (
    TITLE_SIZE,
    save_accessible_svg,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


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
    ordered_panel = ordered_panel_rows(panel_rows)
    display_labels = [_selected_label(row) for row in ordered_panel]
    panel_policy_ids = [str(row.get("policy_id") or "") for row in ordered_panel]
    fig, (ax, context_ax) = plt.subplots(
        1,
        2,
        figsize=(11.8, 6.1),
        gridspec_kw={"width_ratios": (1.35, 1.0), "wspace": 0.30},
    )
    within_policy_mask = np.asarray(
        [[left_policy != right_policy for right_policy in panel_policy_ids] for left_policy in panel_policy_ids],
        dtype=bool,
    )
    masked_position_matrix = np.ma.array(position_matrix, mask=within_policy_mask)
    colormap = plt.get_cmap("Blues").copy()
    colormap.set_bad("#F0F2F4")
    image = ax.imshow(
        masked_position_matrix,
        aspect="equal",
        interpolation="nearest",
        cmap=colormap,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xticks(list(range(len(display_labels))))
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(list(range(len(display_labels))))
    ax.set_yticklabels(display_labels, fontsize=10)
    for row_index, values in enumerate(position_matrix):
        for col_index, value in enumerate(values):
            if within_policy_mask[row_index, col_index]:
                continue
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.3,
                color="#24292f" if value < 0.55 else "#ffffff",
            )
    boundaries = [
        index + 0.5
        for index in range(len(panel_policy_ids) - 1)
        if panel_policy_ids[index] != panel_policy_ids[index + 1]
    ]
    for boundary in boundaries:
        ax.axvline(boundary, color="#57606A", linewidth=1.0, alpha=0.70)
        ax.axhline(boundary, color="#57606A", linewidth=1.0, alpha=0.70)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=30)
    ax.text(
        0.5,
        1.01,
        "Only within-group distances are shown; open position sets differ between groups",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.0,
        color="#57606A",
    )
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, shrink=0.74, pad=0.03)
    distance_label = r"Position-set distance  $d_J = 1 - |A \cap B| / |A \cup B|$"
    cbar.set_label(distance_label, fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    distance_context = build_within_policy_position_distance_context(
        panel_rows=panel_rows,
        candidate_rows=candidate_rows,
        triage_rows=triage_rows,
    )
    legend_handles: list[Line2D] = []
    for policy_id in POLICY_ORDER:
        policy_context = distance_context.get(policy_id)
        if policy_context is None:
            continue
        candidate_distances = np.sort(np.asarray(policy_context["candidate_pair_distances"], dtype=float))
        cumulative_fraction = np.arange(1, candidate_distances.size + 1, dtype=float) / candidate_distances.size
        color = policy_color(policy_id)
        context_ax.step(
            candidate_distances,
            cumulative_fraction,
            where="post",
            color=color,
            linewidth=2.1,
        )
        selected_distances = np.asarray(policy_context["selected_pair_distances"], dtype=float)
        selected_percentiles = np.searchsorted(candidate_distances, selected_distances, side="right") / len(
            candidate_distances
        )
        context_ax.scatter(
            selected_distances,
            selected_percentiles,
            s=46,
            facecolor="#FFFFFF",
            edgecolor=color,
            linewidth=1.8,
            zorder=4,
        )
        legend_handles.append(Line2D([0], [0], color=color, linewidth=2.1, label=policy_label(policy_id)))
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#FFFFFF",
            markeredgecolor="#24292F",
            markeredgewidth=1.5,
            label="Selected pair",
        )
    )
    context_ax.set_xlim(0.0, 1.0)
    context_ax.set_ylim(0.0, 1.02)
    context_ax.set_xlabel(distance_label, fontsize=12.5)
    context_ax.set_ylabel("Cumulative fraction of pairs", fontsize=12.5)
    context_ax.set_title("Selected pairs within each design group", fontsize=TITLE_SIZE - 1, pad=12)
    context_ax.spines[["top", "right"]].set_visible(False)
    context_ax.tick_params(labelsize=11)
    context_ax.grid(color="#D0D7DE", alpha=0.40, linewidth=0.7)
    context_ax.legend(handles=legend_handles, frameon=False, fontsize=10.3, loc="lower right")
    fig.subplots_adjust(left=0.10, right=0.985, top=0.88, bottom=0.20)
    path = plot_root / "selection_mutation_set_dissimilarity.svg"
    matrix_size = len(labels)
    alt = (
        f"{matrix_size}-by-{matrix_size} heatmap of within-group mutation-position dissimilarity among selected "
        "Eco1 RT candidates beside separate cumulative distributions for distal, peripheral, and combined "
        "candidate pairs. Cross-group heatmap cells are suppressed because those groups expose different positions. "
        "Open circles locate selected pairs within the matching generated group."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_mutation_set_dissimilarity",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Shows how much selected sequences reuse the same mutated positions and places each selected pair in the "
            "matching distal, peripheral, or combined candidate distribution. The first pair in each group is chosen "
            "by exhaustive comparison; a third row maximizes its minimum distance from that pair. Exact-substitution "
            "distance is a later tie-break and is reported in the panel table. Cross-group cells are omitted because "
            "the groups expose different positions."
        ),
        interpretation_limit=(
            "Mutation-set dissimilarity reduces overlap but does not guarantee independent mechanisms or provide "
            "functional evidence."
        ),
        render_mode="compact_wide_visual",
    )


def build_within_policy_position_distance_context(
    *,
    panel_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    triage_rows: list[dict[str, object]],
) -> dict[str, dict[str, list[float]]]:
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
    context = {
        policy_id: {
            "candidate_pair_distances": _pairwise_distances(candidate_sets),
            "selected_pair_distances": _pairwise_distances(selected_by_policy.get(policy_id, [])),
        }
        for policy_id, candidate_sets in contract_by_policy.items()
    }
    if not any(values["candidate_pair_distances"] for values in context.values()):
        raise ValueError("Mutation-distance context requires at least one contract-pass within-policy pair")
    if not any(values["selected_pair_distances"] for values in context.values()):
        raise ValueError("Mutation-distance context requires at least one selected within-policy pair")
    return context


def _pairwise_distances(rows: list[frozenset[int]]) -> list[float]:
    return [
        jaccard_distance(rows[left_index], rows[right_index])
        for left_index in range(len(rows))
        for right_index in range(left_index + 1, len(rows))
    ]


def _selected_label(row: dict[str, object]) -> str:
    return short_selected_variant(row)


__all__ = [
    "build_within_policy_position_distance_context",
    "build_selected_mutation_dissimilarity_matrices",
    "write_selected_mutation_dissimilarity_plot",
]
