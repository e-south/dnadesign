"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/pwm/visual_layers.py

BaseRender visual-layer metadata for bidirectional TetR PWM trim sequence rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence

from ..contracts.plan import PwmTrimPanel
from .trim_annotations import retained_span_bracket, trim_edge_markers

PANEL_COLOR = "#3E927F"
EXCLUDED_COLOR = "#C8CED8"
FULL_SITE_BACKDROP_COLOR = "#EEF2F6"
TRIM_CUT_COLOR = "#6B7280"


def sequence_row_visual_meta(
    *,
    complement_sequence: str,
    included: Sequence[int],
    trimmed: Sequence[int],
    panel: PwmTrimPanel,
    site_start: int,
    site_end: int,
) -> dict[str, object]:
    return {
        "complement_sequence": complement_sequence,
        "dim_base_indices": {"primary": trimmed, "complement": trimmed},
        "base_highlight_colors": {
            "primary": {index: PANEL_COLOR for index in included},
            "complement": {index: PANEL_COLOR for index in included},
        },
        "base_dim_color": EXCLUDED_COLOR,
        "span_backdrops": [_full_site_backdrop(site_start=site_start, site_end=site_end)],
        "span_edge_markers": trim_edge_markers(panel, color=TRIM_CUT_COLOR),
        "span_brackets": [retained_span_bracket(color=PANEL_COLOR)],
    }


def _full_site_backdrop(*, site_start: int, site_end: int) -> dict[str, object]:
    return {
        "start": site_start,
        "end": site_end,
        "fill": FULL_SITE_BACKDROP_COLOR,
        "alpha": 0.72,
        "corner_radius": 4.0,
        "cover_rows": "both",
        "edge_color": "#D6DCE6",
        "edge_alpha": 0.75,
        "edge_linewidth": 0.65,
    }


__all__ = ["EXCLUDED_COLOR", "PANEL_COLOR", "sequence_row_visual_meta"]
