"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/view_styles.py

Display-title and style policy for payload-centric YIU views.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass

from dnadesign.baserender import cruncher_showcase_style_overrides
from dnadesign.cruncher.yiu.domain_models import NormalizedPayload

_YIU_FIGURE_SCALE = 1.24
_YIU_VISUAL_SYSTEM_NAME = "bench_strip"


def _pretty_label(text: str | None) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    normalized = re.sub(r"[_-]+", " ", raw)
    return " ".join(token[:1].upper() + token[1:] for token in normalized.split())


def build_payload_view_title(normalized: NormalizedPayload) -> str:
    motif_tfs = sorted({motif.tf_name for motif in normalized.motif_context.motifs if str(motif.tf_name).strip()})
    if len(motif_tfs) == 1:
        tf_label = _pretty_label(motif_tfs[0])
        motif_count = len(normalized.motif_context.motifs)
        suffix = f" ({motif_count} sites)" if motif_count > 1 else ""
        return f"{tf_label} payload{suffix}"
    if normalized.payload_label:
        return _pretty_label(normalized.payload_label)
    return _pretty_label(normalized.name) or "Payload"


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
        system_name=_YIU_VISUAL_SYSTEM_NAME,
        design_note="Dense operator-first evidence row for payload truth, mismatches, and PWM overlays.",
        style_overrides=_evidence_ribbon_style_overrides(),
    ),
    "split_payload": YiuViewStyleProfile(
        view_id="split_payload",
        direction_name="operator_strip",
        system_name=_YIU_VISUAL_SYSTEM_NAME,
        design_note="Lean assembly strip that keeps split-fragment geometry readable without payload-row ornament.",
        style_overrides={**_operator_strip_style_overrides(), "legend": False},
    ),
    "assembled_payload": YiuViewStyleProfile(
        view_id="assembled_payload",
        direction_name="operator_strip",
        system_name=_YIU_VISUAL_SYSTEM_NAME,
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
    "build_payload_view_title",
    "build_yiu_style_overrides",
    "YiuViewStyleProfile",
    "get_yiu_style_profile",
]
