"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/integrations/promoter_panel/style.py

Define style data for the promoter sequence-panel profile.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from copy import deepcopy

_STYLE: dict[str, object] = {
    "figure_scale": 1.60,
    "padding_y": 24.0,
    "overlay_align": "center",
    "layout": {"outer_pad_cells": 0.62},
    "sequence": {
        "strand_gap_cells": 0.18,
        "to_kmer_gap_cells": 0.38,
        "bold_consensus_bases": True,
        "non_consensus_color": "#9CA3AF",
        "tone_quantile_low": 0.10,
        "tone_quantile_high": 0.90,
    },
    "palette": {
        "tf:background": "#C3CAD3",
        "tf:lexA": "#5DADE2",
        "tf:cpxR": "#2D9B66",
        "tf:baeR": "#E58A2B",
        "promoter:sigma70_core:upstream": "#7D86D1",
        "promoter:sigma70_core:downstream": "#C886D1",
    },
    "connectors": False,
    "legend": True,
    "legend_mode": "bottom",
    "legend_inline_side": "auto",
    "legend_inline_margin_cells": 0.28,
    "legend_height_px": 136.0,
    "legend_pad_px": 36.0,
    "legend_content_gap_px": 18.0,
    "legend_patch_w": 88.0,
    "legend_patch_h": 34.0,
    "legend_font_size": 24,
    "legend_gap_patch_text": 22.0,
    "legend_gap_x": 44.0,
    "legend_vertical_align": 1.0,
    "kmer": {
        "box_height_cells": 1.12,
        "fill_alpha": 0.94,
        "text_y_nudge_cells": 0.0,
        "to_logo_gap_cells": 0.12,
    },
    "motif_logo": {
        "layout": "stack",
        "lane_mode": "follow_feature_track",
        "display_mode": "information",
        "height_bits": 2.0,
        "bits_to_cells": 1.35,
        "y_pad_cells": 0.0,
        "letter_x_pad_frac": 0.06,
        "alpha_other": 0.80,
        "alpha_observed": 1.0,
        "debug_bounds": False,
        "letter_coloring": {
            "mode": "match_window_seq",
            "other_color": "#D1D5DB",
            "observed_color_source": "feature_fill",
        },
        "scale_bar": {"enabled": True, "location": "left_of_logo"},
    },
    "show_reverse_complement": True,
    "font_size_seq": 24,
    "font_size_label": 24,
    "font_size_feature_label": 24,
    "font_size_annotation_label": 24,
    "font_size_span_link_label": 24,
    "span_link_line_width": 3.2,
    "span_link_tick_line_width": 2.8,
    "uniform_display_font_size": True,
    "overlay_vertical_anchor": "content_top",
}


def promoter_compact_slide_style() -> dict[str, object]:
    return deepcopy(_STYLE)


__all__ = ["promoter_compact_slide_style"]
