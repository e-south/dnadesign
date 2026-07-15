"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/communication_visuals/constraint_map.py

Compact conservation and design-space map for scientific communication.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
    DISTAL_SCAFFOLD_POLICY_ID,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    CONSERVATION_CLADE9_PROFILE_ID,
    SECTION_CONSTRAINT_EVIDENCE,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.shared.rendering import (
    LABEL_SIZE,
    LEGEND_SIZE,
    TICK_SIZE,
    save_accessible_svg,
    style_open_axes,
)

from .catalog import COMMUNICATION_ROLE, DESIGN_SPACE_MAP_ID
from .style import (
    CONSERVATION_COLOR,
    CONTACT_COLOR,
    MOTIF_COLOR,
    POLICY_COLORS,
    RECOGNITION_COLOR,
    THUMB_COLOR,
)

_FILE_NAME = "design_space_map.svg"
_MOTIF_DISPLAY_LABELS = {"naxxh": "NAxxH", "yadd": "YADD", "vtg": "VTG"}
_ANGSTROM = "\N{LATIN CAPITAL LETTER A WITH RING ABOVE}"
_GREATER_THAN_OR_EQUAL = "\N{GREATER-THAN OR EQUAL TO}"
_LESS_THAN_OR_EQUAL = "\N{LESS-THAN OR EQUAL TO}"


def write_design_space_map(
    *,
    panel_root: Path,
    conservation_rows: list[dict[str, Any]],
    policy_position_rows: list[dict[str, Any]],
    mask_residues: list[dict[str, Any]],
    mask_set_path: Path,
    conservation_profile_path: Path,
    policy_positions_path: Path,
) -> dict[str, Any]:
    """Render conservation evidence and fixed/open policy tracks on one residue axis."""

    path = panel_root / _FILE_NAME
    clade_rows = [
        row for row in conservation_rows if str(row.get("profile_id") or "") == CONSERVATION_CLADE9_PROFILE_ID
    ]
    positions = sorted({int(row["eco1_position"]) for row in policy_position_rows})
    if not positions:
        raise ValueError("Communication design-space map requires generation-policy positions")
    position_rows = _position_context_rows(policy_position_rows)
    conservation_by_position = {int(row["canonical_position"]): row for row in clade_rows}
    motif_segments = _motif_anchor_segments(mask_residues)
    tracks = (
        (
            f"Fixed: WT is clade-9 plurality at {_GREATER_THAN_OR_EQUAL}25%",
            _positions_with_value(position_rows, "is_conserved_core"),
            CONSERVATION_COLOR,
            (),
        ),
        (
            "Fixed motif context windows (study choice)",
            _positions_with_value(position_rows, "motif_context_codes"),
            MOTIF_COLOR,
            motif_segments,
        ),
        (
            f"Direct DNA/RNA contacts {_LESS_THAN_OR_EQUAL}5 {_ANGSTROM} (Wang et al.; 7V9U)",
            _positions_with_value(position_rows, "is_direct_contact_le_5a"),
            CONTACT_COLOR,
            (),
        ),
        (
            "Thumb-contact residues (Wang et al.)",
            _positions_with_value(position_rows, "is_wang_thumb_track"),
            THUMB_COLOR,
            (),
        ),
        (
            "Primer-RNA recognition 255-311 (Inouye et al.)",
            _positions_with_value(position_rows, "is_c_terminal_thumb_context"),
            RECOGNITION_COLOR,
            (),
        ),
        (
            f"Open: distal scaffold >10 {_ANGSTROM}",
            _open_positions(policy_position_rows, DISTAL_SCAFFOLD_POLICY_ID),
            POLICY_COLORS[DISTAL_SCAFFOLD_POLICY_ID],
            (),
        ),
        (
            f"Open: peripheral shell >5 to {_LESS_THAN_OR_EQUAL}10 {_ANGSTROM}",
            _open_positions(policy_position_rows, NEAR_DNA_RNA_ACID_FREE_POLICY_ID),
            POLICY_COLORS[NEAR_DNA_RNA_ACID_FREE_POLICY_ID],
            (),
        ),
        (
            "Open: combined distal + peripheral",
            _open_positions(policy_position_rows, COMBINED_NEAR_PLUS_DISTAL_POLICY_ID),
            POLICY_COLORS[COMBINED_NEAR_PLUS_DISTAL_POLICY_ID],
            (),
        ),
    )

    fig = plt.figure(figsize=(14.8, 5.25))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.55, 2.65), hspace=0.13)
    conservation_ax = fig.add_subplot(grid[0])
    track_ax = fig.add_subplot(grid[1], sharex=conservation_ax)
    frequencies = [
        float(conservation_by_position.get(position, {}).get("wt_frequency") or 0.0) for position in positions
    ]
    conservation_ax.plot(positions, frequencies, color=CONSERVATION_COLOR, linewidth=1.25)
    conservation_ax.fill_between(positions, frequencies, color=CONSERVATION_COLOR, alpha=0.10)
    conservation_ax.axhline(0.25, color="#6E7781", linestyle="--", linewidth=1.0)
    conservation_ax.text(
        positions[-1] - 4,
        0.29,
        "Declared\n25% rule",
        ha="right",
        va="bottom",
        fontsize=LEGEND_SIZE,
        color="#57606A",
        linespacing=0.9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.2},
    )
    conservation_ax.set_ylabel("WT residue frequency\nin clade-9 MSA", fontsize=LABEL_SIZE, labelpad=12)
    conservation_ax.set_ylim(-0.02, 1.04)
    conservation_ax.tick_params(axis="x", labelbottom=False)
    style_open_axes(conservation_ax)

    track_ax.axhspan(-0.5, 4.5, color="#66707A", alpha=0.045, zorder=-3)
    track_ax.axhspan(4.5, 7.5, color="#2C7A5B", alpha=0.045, zorder=-3)
    track_ax.axhline(4.5, color="#9AA4AE", linewidth=0.8, alpha=0.70, zorder=-2)
    for track_index, (_label, active_positions, color, segment_labels) in enumerate(tracks):
        for start, end in _contiguous_runs(active_positions):
            track_ax.add_patch(
                Rectangle(
                    (start - 0.5, track_index - 0.25),
                    end - start + 1,
                    0.50,
                    facecolor=color,
                    edgecolor="none",
                    zorder=1,
                )
            )
        for segment_label, segment_positions in segment_labels:
            for start, end in _contiguous_runs(segment_positions):
                track_ax.add_patch(
                    Rectangle(
                        (start - 0.5, track_index - 0.34),
                        end - start + 1,
                        0.62,
                        facecolor=color,
                        edgecolor="#FFFFFF",
                        linewidth=0.8,
                        zorder=2,
                    )
                )
                track_ax.text(
                    min(end + 2.0, positions[-1] - 1),
                    track_index,
                    segment_label,
                    ha="left" if end + 2.0 <= positions[-1] - 1 else "right",
                    va="center",
                    color=color,
                    fontsize=9.5,
                    weight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.6},
                    zorder=3,
                )
    track_ax.set_yticks(
        range(len(tracks)),
        [label for label, _positions, _color, _segment_labels in tracks],
        fontsize=TICK_SIZE,
    )
    track_ax.set_ylim(len(tracks) - 0.45, -0.55)
    track_ax.set_xlim(positions[0] - 0.5, positions[-1] + 0.5)
    track_ax.set_xlabel("Eco1 RT residue position", fontsize=LABEL_SIZE)
    track_ax.spines[["top", "right", "left"]].set_visible(False)
    track_ax.tick_params(axis="y", length=0)
    track_ax.grid(axis="x", color="#D8DEE4", alpha=0.45, linewidth=0.7)
    track_ax.set_axisbelow(True)
    plot_left = 0.31
    plot_right = 0.985
    fig.text(
        (plot_left + plot_right) / 2,
        0.975,
        "Conservation and structure define the redesign space",
        ha="center",
        va="top",
        fontsize=18,
    )
    fig.subplots_adjust(left=plot_left, right=plot_right, bottom=0.145, top=0.86)

    alt_text = (
        "Landscape residue-position figure showing clade-9 WT frequency above eight evidence and design-space "
        "tracks. The declared fixed motif context windows are 99-115, 189-204, and 237-251; exact "
        "literature-annotated NAxxH, YADD, and VTG "
        "anchors at 105-109, 195-198, and 243-245 are labeled as darker slivers. "
        "Wang/7V9U-derived "
        "contact tracks and "
        "the Inouye primer-RNA recognition region are shown separately from distal, peripheral, and combined open "
        "positions."
    )
    save_accessible_svg(fig, path, title="Conservation and structure define the redesign space", description=alt_text)
    return make_deliverable_row(
        deliverable_id=DESIGN_SPACE_MAP_ID,
        section=SECTION_CONSTRAINT_EVIDENCE,
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=[
            "conservation_profile.parquet",
            "mask_set.yaml",
            "generation_policies_v3/generation_policy_positions.parquet",
        ],
        input_hashes=file_hashes(
            {
                "conservation_profile": conservation_profile_path,
                "mask_set": mask_set_path,
                "generation_policy_positions": policy_positions_path,
            }
        ),
        alt_text=alt_text,
        description=(
            "Separates each reason for fixing a residue from the three designable position sets. The clade-9 MSA "
            "track is adjacent to the WT-frequency profile and states the declared 25% plurality rule."
        ),
        interpretation_limit=(
            "The motif-context widths, 25% plurality rule, and 5 A/10 A distance boundaries are declared "
            "design choices, not functional discontinuities. The named motif anchors are narrower than their fixed "
            "context windows."
        ),
        title="Conservation and structure define the redesign space",
        role=COMMUNICATION_ROLE,
        render_mode="compact_wide_visual",
        method_summary=(
            "WT frequency comes from the accepted clade-9 alignment. Binary tracks are read from the active "
            "generation-policy position manifest so overlapping protection reasons remain visible."
        ),
        evidence_summary={
            "residue_positions": len(positions),
            "clade9_alignment_positions": len(clade_rows),
            "displayed_tracks": len(tracks),
        },
    )


def _position_context_rows(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    preferred = [row for row in rows if str(row.get("policy_id") or "") == COMBINED_NEAR_PLUS_DISTAL_POLICY_ID]
    source = preferred or rows
    return {int(row["eco1_position"]): row for row in source}


def _positions_with_value(rows: dict[int, dict[str, Any]], field: str) -> set[int]:
    return {position for position, row in rows.items() if bool(row.get(field))}


def _motif_anchor_segments(mask_residues: list[dict[str, Any]]) -> tuple[tuple[str, set[int]], ...]:
    reason_positions: dict[str, set[int]] = {}
    for row in mask_residues:
        position = int(row["canonical_position"])
        for reason in str(row.get("manual_mask_reason") or "").split(";"):
            normalized = reason.strip().lower()
            if normalized:
                reason_positions.setdefault(normalized, set()).add(position)
    return tuple(
        (display_label, reason_positions.get(reason_code, set()))
        for reason_code, display_label in (
            ("retron_x_naxxh", _MOTIF_DISPLAY_LABELS["naxxh"]),
            ("catalytic_yadd", _MOTIF_DISPLAY_LABELS["yadd"]),
            ("retron_y_vtg", _MOTIF_DISPLAY_LABELS["vtg"]),
        )
    )


def _open_positions(rows: list[dict[str, Any]], policy_id: str) -> set[int]:
    return {
        int(row["eco1_position"])
        for row in rows
        if str(row.get("policy_id") or "") == policy_id and bool(row.get("is_open_position"))
    }


def _contiguous_runs(values: Iterable[int]) -> list[tuple[int, int]]:
    ordered = sorted(set(int(value) for value in values))
    if not ordered:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        runs.append((start, previous))
        start = previous = value
    runs.append((start, previous))
    return runs
