"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/showcase_style.py

Canonical style helpers for curated showcase-like Sequence Rows renders.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

_CRUNCHER_SHOWCASE_STYLE_OVERRIDES: dict[str, object] = {
    "figure_scale": 1.60,
    "padding_y": 24.0,
    "overlay_align": "center",
    "layout": {
        "outer_pad_cells": 0.20,
    },
    "sequence": {
        "strand_gap_cells": 0.18,
        "to_kmer_gap_cells": 0.12,
        "bold_consensus_bases": True,
        "non_consensus_color": "#9CA3AF",
        "tone_quantile_low": 0.10,
        "tone_quantile_high": 0.90,
    },
    "palette": {
        "tf:lexA": "#B45309",
        "tf:cpxR": "#0F766E",
    },
    "connectors": False,
    "legend_mode": "inline",
    "legend_inline_side": "auto",
    "legend_inline_margin_cells": 0.28,
    "legend_font_size": 10,
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
        "scale_bar": {
            "enabled": True,
            "location": "left_of_logo",
        },
    },
}


def cruncher_showcase_style_overrides() -> Mapping[str, object]:
    """Return canonical Cruncher showcase style overrides (defensive copy)."""
    return deepcopy(_CRUNCHER_SHOWCASE_STYLE_OVERRIDES)


__all__ = ["cruncher_showcase_style_overrides"]
