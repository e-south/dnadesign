"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/msa_panel_layout.py

Layout helpers for Eco1 review-deliverable MSA panels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    CONSERVATION_SUBTYPE_PROFILE_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.rendering import (
    OKABE_ITO,
)

_TRACK_50_Y = -5.25
_TRACK_25_Y = -3.65
_TRACK_PROTECTED_Y = -2.05
_TRACK_HEIGHT = 0.8
_TRACK_TOP_Y_LIMIT = -5.75
_TRACK_50_TICK = -4.85
_TRACK_25_TICK = -3.25
_TRACK_PROTECTED_TICK = -1.65
_BASELINE_FIXED_LABEL = "Baseline fixed residues (clade 9 p25 + 5 A)"
_Y_TICK_SIZE_STANDARD = 8.2
_Y_TICK_SIZE_COMPACT = 6.2
_Y_TICK_SIZE_DENSE = 4.8


def _track_tick_labels(panel_profile: Any) -> tuple[tuple[float, str], ...]:
    scope = _short_scope_label(panel_profile)
    return (
        (_TRACK_50_TICK, f"{scope} 50% WT plurality"),
        (_TRACK_25_TICK, f"{scope} 25% WT plurality"),
        (_TRACK_PROTECTED_TICK, _BASELINE_FIXED_LABEL),
    )


def _short_scope_label(panel_profile: Any) -> str:
    if panel_profile.profile_id == CONSERVATION_SUBTYPE_PROFILE_ID:
        return "II-A3/42_1"
    return "clade 9"


def _style_y_tick_labels(
    ax: Any,
    *,
    selected_records: list[tuple[str, str]],
    subtype_record_ids: set[str],
    row_label_size: float,
    track_label_count: int,
) -> None:
    for index, label in enumerate(ax.get_yticklabels()):
        label.set_fontsize(row_label_size)
        if index < track_label_count:
            label.set_color("#333333")
            continue
        record_index = index - track_label_count
        if selected_records[record_index][0] in subtype_record_ids:
            label.set_bbox(
                {
                    "facecolor": OKABE_ITO["sky"],
                    "edgecolor": "none",
                    "alpha": 0.18,
                    "pad": 1.1,
                }
            )


def _figure_size(row_count: int) -> tuple[float, float]:
    return _figure_width(row_count), _figure_height(row_count)


def _figure_width(row_count: int) -> float:
    if row_count <= 80:
        return 18.4
    return 22.4


def _figure_height(row_count: int) -> float:
    if row_count <= 60:
        return max(5.8, row_count * 0.22 + 1.75)
    if row_count <= 140:
        return row_count * 0.14 + 2.15
    return row_count * 0.096 + 2.35


def _figure_margins(row_count: int, *, fig_height: float) -> dict[str, float]:
    bottom = max(0.09, min(0.18, 1.08 / fig_height))
    top_margin_floor = 0.052 if row_count > 140 else 0.036
    top = 1.0 - max(top_margin_floor, min(0.12, 0.78 / fig_height))
    left = 0.205 if row_count <= 80 else 0.112
    return {
        "left": left,
        "right": 0.995,
        "bottom": bottom,
        "top": top,
    }


def _axes_center_x(margins: dict[str, float]) -> float:
    return (float(margins["left"]) + float(margins["right"])) / 2.0


def _subtype_fill_left_extension(
    *,
    row_labels: list[str],
    row_label_size: float,
    position_count: int,
    fig_width: float,
    axes_width_fraction: float,
) -> float:
    if not row_labels or position_count <= 0:
        return 0.0
    max_label_chars = max(len(label) for label in row_labels)
    approximate_label_points = max_label_chars * row_label_size * 0.34
    axes_width_points = max(fig_width * 72.0 * axes_width_fraction, 1.0)
    points_per_position = axes_width_points / float(position_count)
    return max(3.0, min(18.0, approximate_label_points / points_per_position))


def _msa_y_tick_size(row_count: int) -> float:
    if row_count <= 60:
        return _Y_TICK_SIZE_STANDARD
    if row_count <= 140:
        return _Y_TICK_SIZE_COMPACT
    return _Y_TICK_SIZE_DENSE


def _position_tick_indexes(positions: list[int]) -> list[int]:
    return [index for index, position in enumerate(positions) if position == 1 or position % 40 == 0]


def _residue_label_size(position_count: int, row_count: int) -> float:
    if row_count <= 80:
        return 7.4 if position_count <= 180 else 6.4
    return 6.0
