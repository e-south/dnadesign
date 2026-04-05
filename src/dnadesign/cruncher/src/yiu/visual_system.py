"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/visual_system.py

Named visual-system policy for payload-centric YIU bundle views.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from dnadesign.baserender import cruncher_showcase_style_overrides

YIU_VISUAL_SYSTEM_NAME = "bench_strip"
_YIU_FIGURE_SCALE = 1.24


@dataclass(frozen=True)
class YiuViewStyleProfile:
    view_id: str
    direction_name: str
    system_name: str
    design_note: str
    style_overrides: dict[str, object]


def _operator_strip_style_overrides() -> dict[str, object]:
    return {
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


def _evidence_ribbon_style_overrides() -> dict[str, object]:
    base = dict(cruncher_showcase_style_overrides())
    base.update(
        {
            "figure_scale": _YIU_FIGURE_SCALE,
            "padding_x": 42.0,
            "padding_y": 24.0,
            "track_spacing": 20.0,
            "font_size_seq": 13,
            "font_size_label": 11,
            "legend": False,
            "legend_mode": "none",
            "connectors": True,
            "connector_width": 1.1,
            "connector_alpha": 0.78,
            "connector_dash": (),
        }
    )
    return base


_STYLE_PROFILES: dict[str, YiuViewStyleProfile] = {
    "payload": YiuViewStyleProfile(
        view_id="payload",
        direction_name="evidence_ribbon",
        system_name=YIU_VISUAL_SYSTEM_NAME,
        design_note="Dense operator-first evidence row for payload truth, mismatches, and PWM overlays.",
        style_overrides=_evidence_ribbon_style_overrides(),
    ),
    "split_payload": YiuViewStyleProfile(
        view_id="split_payload",
        direction_name="operator_strip",
        system_name=YIU_VISUAL_SYSTEM_NAME,
        design_note="Lean assembly strip that keeps split-fragment geometry readable without payload-row ornament.",
        style_overrides={**_operator_strip_style_overrides(), "legend": False},
    ),
    "assembled_payload": YiuViewStyleProfile(
        view_id="assembled_payload",
        direction_name="operator_strip",
        system_name=YIU_VISUAL_SYSTEM_NAME,
        design_note="Lean reassembly strip that centers the restored payload order and junction context.",
        style_overrides={**_operator_strip_style_overrides(), "legend": False, "padding_y": 28.0},
    ),
}


def get_yiu_style_profile(view_id: str) -> YiuViewStyleProfile:
    try:
        profile = _STYLE_PROFILES[view_id]
    except KeyError as exc:
        supported = ", ".join(sorted(_STYLE_PROFILES))
        raise ValueError(f"unsupported YIU view id {view_id!r}; expected one of: {supported}") from exc
    return YiuViewStyleProfile(
        view_id=profile.view_id,
        direction_name=profile.direction_name,
        system_name=profile.system_name,
        design_note=profile.design_note,
        style_overrides=deepcopy(profile.style_overrides),
    )


def build_yiu_style_overrides(view_id: str) -> dict[str, object]:
    return get_yiu_style_profile(view_id).style_overrides


__all__ = [
    "build_yiu_style_overrides",
    "get_yiu_style_profile",
    "YIU_VISUAL_SYSTEM_NAME",
    "YiuViewStyleProfile",
]
