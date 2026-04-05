"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/visual_foundations.py

Producer-owned style foundations for the payload-centric YIU visual system.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping


def _bench_strip_palette() -> dict[str, str]:
    return {
        "tf:acrR": "#4C8E7F",
        "tf:araC": "#B86E8D",
        "tf:baeR": "#5E84B8",
        "tf:lexA": "#A87644",
        "tf:cpxR": "#3E927F",
        "tf:fnr": "#B89756",
        "tf:fur": "#5B98A5",
        "tf:lacI": "#6B84B8",
        "tf:lrp": "#789A63",
        "tf:rcdA": "#A29070",
        "tf:soxR": "#B66F8C",
        "tf:soxS": "#BC8466",
    }


def _bench_strip_sequence_style() -> dict[str, object]:
    return {
        "strand_gap_cells": 0.16,
        "to_kmer_gap_cells": 0.10,
        "bold_consensus_bases": True,
        "non_consensus_color": "#9AA4B3",
        "tone_quantile_low": 0.10,
        "tone_quantile_high": 0.90,
    }


def _bench_strip_kmer_style() -> dict[str, object]:
    return {
        "box_height_cells": 1.08,
        "fill_alpha": 0.94,
        "text_y_nudge_cells": 0.0,
        "to_logo_gap_cells": 0.10,
    }


def _bench_strip_motif_logo_style() -> dict[str, object]:
    return {
        "layout": "stack",
        "lane_mode": "follow_feature_track",
        "display_mode": "information",
        "height_bits": 2.0,
        "bits_to_cells": 1.28,
        "y_pad_cells": 0.0,
        "letter_x_pad_frac": 0.05,
        "alpha_other": 0.82,
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
    }


_BENCH_STRIP_FOUNDATION: dict[str, object] = {
    "padding_y": 26.0,
    "overlay_align": "center",
    "layout": {"outer_pad_cells": 0.22},
    "sequence": _bench_strip_sequence_style(),
    "palette": _bench_strip_palette(),
    "connectors": False,
    "legend_mode": "inline",
    "legend_inline_side": "auto",
    "legend_inline_margin_cells": 0.24,
    "legend_font_size": 10,
    "kmer": _bench_strip_kmer_style(),
    "motif_logo": _bench_strip_motif_logo_style(),
}


def bench_strip_style_foundation() -> Mapping[str, object]:
    """Return the producer-owned YIU bench-strip style seed."""
    return deepcopy(_BENCH_STRIP_FOUNDATION)


__all__ = ["bench_strip_style_foundation"]
