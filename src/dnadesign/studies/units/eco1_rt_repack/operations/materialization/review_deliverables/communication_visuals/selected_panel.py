"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/communication_visuals/selected_panel.py

Compact selected-panel flow and mutation map for communication.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_PANEL_SELECTION,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.plot_support import (
    canonical_mutations,
    parse_mutation,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.shared.rendering import (
    LABEL_SIZE,
    TICK_SIZE,
    save_accessible_svg,
)

from .catalog import COMMUNICATION_ROLE, SELECTED_PANEL_ID
from .style import GRID_COLOR, POLICY_ORDER, TEXT_COLOR, policy_color, policy_label

_FILE_NAME = "selected_panel.svg"
_COLUMN_HEADER_Y = -0.62


def write_selected_panel(
    *,
    panel_root: Path,
    triage_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    policy_position_rows: list[dict[str, Any]],
    triage_table_path: Path,
    selection_panel_path: Path,
    policy_positions_path: Path,
) -> dict[str, Any]:
    """Render candidate counts and the selected mutation geography in one landscape figure."""

    if not selected_rows:
        raise ValueError("Communication selected-panel figure requires selected rows")
    path = panel_root / _FILE_NAME
    ordered_rows = sorted(
        selected_rows,
        key=lambda row: (
            _policy_index(str(row.get("policy_id") or "")),
            int(row.get("within_group_rank") or row.get("selection_rank") or 999),
            str(row.get("candidate_id") or ""),
        ),
    )
    max_position = max(int(row["eco1_position"]) for row in policy_position_rows)
    contract_rows = [row for row in triage_rows if bool(row.get("selection_contract_pass"))]
    geometry_rows = [row for row in triage_rows if str(row.get("hard_gate_status") or "") == "eligible"]
    policy_counts = Counter(str(row.get("policy_id") or "") for row in contract_rows)
    r13_substituted_by_policy = Counter(
        str(row.get("policy_id") or "")
        for row in contract_rows
        if int(row.get("wang_alpha1_r13_mutation_count") or 0) > 0
    )
    fig = plt.figure(figsize=(13.4, 4.75))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(0.82, 3.25),
        width_ratios=(10.2, 2.1),
        hspace=0.16,
        wspace=0.04,
    )
    flow_ax = fig.add_subplot(grid[0, :])
    mutation_ax = fig.add_subplot(grid[1, 0])
    summary_ax = fig.add_subplot(grid[1, 1], sharey=mutation_ax)
    _draw_flow_summary(
        flow_ax,
        accepted_count=len(triage_rows),
        geometry_count=len(geometry_rows),
        policy_counts=policy_counts,
        selected_count=len(ordered_rows),
    )
    _draw_mutation_map(
        mutation_ax,
        summary_ax=summary_ax,
        ordered_rows=ordered_rows,
        max_position=max_position,
    )
    title = "Selected sequences span three intervention levels"
    fig.suptitle(title, fontsize=18)
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.15, top=0.88)

    policy_row_counts = {
        policy_label(policy_id): sum(str(row.get("policy_id") or "") == policy_id for row in ordered_rows)
        for policy_id in POLICY_ORDER
    }
    alt_text = (
        f"Landscape two-part figure. The top summarizes {len(triage_rows)} accepted sequences, "
        f"{len(geometry_rows)} local-geometry-pass sequences and the three design groups. The lower panel maps all "
        f"substitution positions for {len(ordered_rows)} selected rows along the Eco1 RT sequence, with total "
        "mutation count and near-DNA/RNA charge change shown at right."
    )
    save_accessible_svg(
        fig,
        path,
        title=title,
        description=alt_text,
    )
    return make_deliverable_row(
        deliverable_id=SELECTED_PANEL_ID,
        section=SECTION_PANEL_SELECTION,
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=[
            "generation_policies_v3/selection/candidate_triage_table.parquet",
            "generation_policies_v3/selection/candidate_selection_panel.parquet",
            "generation_policies_v3/generation_policy_positions.parquet",
        ],
        input_hashes=file_hashes(
            {
                "candidate_triage_table": triage_table_path,
                "candidate_selection_panel": selection_panel_path,
                "generation_policy_positions": policy_positions_path,
            }
        ),
        alt_text=alt_text,
        description=(
            "Combines the candidate counts needed to understand the selection trace with a residue-position map "
            "of the selected panel. The alpha-1 review remains in the evidence table rather than "
            "adding another visual column."
        ),
        interpretation_limit=(
            "The selected sequences are mutation-set-diverse experimental contrasts, not independent mechanisms "
            "or ranked activity predictions."
        ),
        title=title,
        role=COMMUNICATION_ROLE,
        render_mode="wide_visual",
        method_summary=(
            "Local-geometry-pass rows are counted by generation policy. Selected rows are ordered by group and "
            "within-group rank; every canonical substitution is plotted at its Eco1 residue position."
        ),
        evidence_summary={
            "accepted_sequences": len(triage_rows),
            "local_geometry_pass_sequences": len(geometry_rows),
            "selected_sequences": len(ordered_rows),
            "selected_by_policy": policy_row_counts,
            "r13_substituted_by_policy": {
                policy_label(policy_id): r13_substituted_by_policy[policy_id] for policy_id in POLICY_ORDER
            },
        },
    )


def _draw_flow_summary(
    ax: Any,
    *,
    accepted_count: int,
    geometry_count: int,
    policy_counts: Counter[str],
    selected_count: int,
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        (0.02, 0.16, "ProteinMPNN", "Complete sequences", f"{accepted_count:,}", "#6E7781"),
        (0.23, 0.16, "ColabFold", "Local Cα RMSD ≤2.5 Å", f"{geometry_count:,}", "#56B4E9"),
    ]
    policy_x = (0.47, 0.59, 0.71)
    for x_coord, policy_id in zip(policy_x, POLICY_ORDER, strict=True):
        stages.append(
            (
                x_coord,
                0.09,
                "ProteinMPNN",
                f"{policy_label(policy_id)} policy",
                f"{policy_counts.get(policy_id, 0):,}",
                policy_color(policy_id),
            )
        )
    stages.append(
        (
            0.87,
            0.11,
            "Jaccard distance",
            "Selected panel",
            f"{selected_count}",
            "#B8860B",
        )
    )
    for x_coord, width, method, label, count, color in stages:
        center = x_coord + width / 2
        ax.text(
            center,
            0.74,
            method,
            ha="center",
            va="center",
            fontsize=8.8,
            weight="bold",
            color=TEXT_COLOR,
        )
        ax.text(center, 0.49, label, ha="center", va="center", fontsize=8.8, color=TEXT_COLOR)
        ax.text(center, 0.23, count, ha="center", va="center", fontsize=11.8, weight="bold", color=TEXT_COLOR)
        ax.add_patch(Rectangle((x_coord, 0.06), width, 0.035, facecolor=color, edgecolor="none", alpha=0.9))
    for start, end in ((0.18, 0.23), (0.39, 0.47), (0.80, 0.87)):
        ax.annotate(
            "",
            xy=(end, 0.41),
            xytext=(start, 0.41),
            arrowprops={"arrowstyle": "->", "color": "#6E7781", "linewidth": 1.0},
        )


def _draw_mutation_map(
    ax: Any,
    *,
    summary_ax: Any,
    ordered_rows: list[dict[str, Any]],
    max_position: int,
) -> None:
    for row_index, row in enumerate(ordered_rows):
        policy_id = str(row.get("policy_id") or "")
        mutations = [parse_mutation(value) for value in canonical_mutations(row.get("canonical_mutations"))]
        positions = [int(mutation["position"]) for mutation in mutations]
        if positions:
            ax.vlines(
                positions,
                row_index - 0.25,
                row_index + 0.25,
                color=policy_color(policy_id),
                linewidth=2.2,
                alpha=1.0,
            )
        summary_ax.text(
            0,
            row_index,
            str(int(row.get("mutation_count_total") or len(positions))),
            ha="center",
            va="center",
            fontsize=TICK_SIZE,
        )
        charge = row.get("nucleic_acid_facing_charge_delta")
        charge_text = "NA" if charge is None else f"{float(charge):+g}"
        summary_ax.text(1, row_index, charge_text, ha="center", va="center", fontsize=TICK_SIZE)
    if max_position >= 311:
        ax.axvspan(254.5, 311.5, color="#6E7781", alpha=0.08, zorder=0)
        ax.text(
            283,
            _COLUMN_HEADER_Y,
            "Fixed 255-311",
            ha="center",
            va="center",
            fontsize=9.5,
            color="#57606A",
        )
    labels = [_panel_label(row) for row in ordered_rows]
    ax.set_yticks(range(len(ordered_rows)), labels, fontsize=TICK_SIZE)
    ax.set_ylim(len(ordered_rows) - 0.4, -0.9)
    ax.set_xlim(0.5, max_position + 0.5)
    ax.set_xlabel("Eco1 RT residue position", fontsize=LABEL_SIZE)
    ax.set_ylabel("Selected sequence", fontsize=LABEL_SIZE)
    tick_step = 50 if max_position > 120 else max(1, max_position // 6)
    ax.set_xticks(list(range(1, max_position + 1, tick_step)))
    ax.grid(axis="x", color=GRID_COLOR, alpha=0.55, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_bounds(1, max_position)
    ax.tick_params(labelsize=TICK_SIZE)

    summary_ax.set_xlim(-0.5, 1.5)
    summary_ax.set_xticks([])
    for x_coord, label in ((0, "Mutations"), (1, "Shell charge")):
        summary_ax.text(
            x_coord,
            _COLUMN_HEADER_Y,
            label,
            ha="center",
            va="center",
            fontsize=9.5,
            color="#57606A",
        )
    summary_ax.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
    summary_ax.spines[["top", "right", "bottom"]].set_visible(False)
    summary_ax.spines["left"].set_color("#D0D7DE")
    summary_ax.spines["left"].set_linewidth(0.9)


def _panel_label(row: dict[str, Any]) -> str:
    policy_id = str(row.get("policy_id") or "")
    prefix = {POLICY_ORDER[0]: "D", POLICY_ORDER[1]: "P", POLICY_ORDER[2]: "C"}.get(policy_id, "V")
    rank = int(row.get("within_group_rank") or row.get("selection_rank") or 0)
    return f"{prefix}{rank}"


def _policy_index(policy_id: str) -> int:
    try:
        return POLICY_ORDER.index(policy_id)
    except ValueError:
        return len(POLICY_ORDER)
