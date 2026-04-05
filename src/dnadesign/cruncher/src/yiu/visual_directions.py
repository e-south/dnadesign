"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/visual_directions.py

Named direction deltas for the producer-owned YIU bench-strip visual system.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.yiu.visual_foundations import bench_strip_style_foundation

_YIU_FIGURE_SCALE = 1.24

_COMMON_STRIP_STYLE_OVERRIDES: dict[str, object] = {
    "figure_scale": _YIU_FIGURE_SCALE,
    "padding_x": 42.0,
    "padding_y": 24.0,
    "track_spacing": 20.0,
    "font_size_seq": 13,
    "font_size_label": 11,
    "legend_mode": "none",
    "legend_font_size": 10,
    "legend_gap_x": 10.0,
    "legend_height_px": 52.0,
    "layout": {"outer_pad_cells": 0.18},
    "sequence": {"strand_gap_cells": 0.22, "to_kmer_gap_cells": 0.18},
    "kmer": {"box_height_cells": 1.02, "fill_alpha": 0.94, "text_y_nudge_cells": 0.0},
    "overlay_align": "center",
    "connector_width": 1.1,
    "connector_alpha": 0.78,
    "connector_dash": (),
}


def _merge_style_overrides(base: dict[str, object], **updates: object) -> dict[str, object]:
    merged = dict(base)
    for key, value in updates.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            nested = dict(existing)
            nested.update(value)
            merged[key] = nested
            continue
        merged[key] = value
    return merged


def _bench_strip_direction_base() -> dict[str, object]:
    return _merge_style_overrides(
        dict(bench_strip_style_foundation()),
        **_COMMON_STRIP_STYLE_OVERRIDES,
    )


def operator_strip_style_overrides(*, padding_y: float = 24.0) -> dict[str, object]:
    return _merge_style_overrides(
        _bench_strip_direction_base(),
        padding_y=padding_y,
        legend=False,
        connectors=False,
    )


def evidence_ribbon_style_overrides() -> dict[str, object]:
    return _merge_style_overrides(
        _bench_strip_direction_base(),
        legend=False,
        connectors=True,
    )


__all__ = [
    "evidence_ribbon_style_overrides",
    "operator_strip_style_overrides",
]
