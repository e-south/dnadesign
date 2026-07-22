"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/pwm/trim_annotations.py

Trim-boundary annotation metadata for bidirectional TetR PWM review panels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..contracts.plan import PwmTrimPanel


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


__all__ = ["retained_span_bracket", "trim_edge_markers"]
