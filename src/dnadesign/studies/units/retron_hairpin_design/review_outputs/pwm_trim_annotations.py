"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/pwm_trim_annotations.py

Trim-boundary annotation metadata for bidirectional TetR PWM review panels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .plan import PwmTrimPanel
from .pwm_typography import BOUNDARY_TICK_FONT_SIZE


def trim_edge_markers(panel: PwmTrimPanel, *, color: str) -> list[dict[str, object]]:
    if panel.trim_5p_nt == 0 and panel.trim_3p_nt == 0:
        return []
    return [
        {
            "start": panel.retained_start_0,
            "end": panel.retained_end_0,
            "color": color,
            "alpha": 0.72,
            "linewidth": 1.5,
            "cover_rows": "all",
        }
    ]


def boundary_ticks(panel: PwmTrimPanel, *, color: str) -> list[dict[str, object]]:
    return [
        _boundary_tick(position=panel.retained_start_0, color=color),
        _boundary_tick(position=panel.retained_end_0, color=color),
    ]


def retained_span_bracket(*, color: str) -> dict[str, object]:
    return {
        "target_feature_id": "tetO_retained_payload_span",
        "label": "",
        "color": color,
        "offset_px": 3.0,
        "height_px": 5.5,
        "linewidth": 1.25,
        "font_size": 14,
    }


def _boundary_tick(*, position: int, color: str) -> dict[str, object]:
    return {
        "position": position,
        "label": str(position),
        "emphasis": "active",
        "color": color,
        "font_size": BOUNDARY_TICK_FONT_SIZE,
        "linewidth": 0.8,
    }


__all__ = ["BOUNDARY_TICK_FONT_SIZE", "boundary_ticks", "retained_span_bracket", "trim_edge_markers"]
