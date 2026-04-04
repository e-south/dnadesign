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

from dnadesign.baserender import cruncher_showcase_style_overrides
from dnadesign.cruncher.yiu.domain_models import NormalizedPayload

_YIU_FIGURE_SCALE = 1.24


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


def _base_sequence_style_overrides() -> dict[str, object]:
    return {
        "figure_scale": _YIU_FIGURE_SCALE,
        "padding_x": 42.0,
        "padding_y": 24.0,
        "font_size_seq": 13,
        "font_size_label": 11,
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


def build_yiu_style_overrides(view_id: str) -> dict[str, object]:
    if view_id == "payload":
        base = dict(cruncher_showcase_style_overrides())
        base["figure_scale"] = _YIU_FIGURE_SCALE
        base["padding_x"] = 42.0
        base["padding_y"] = 24.0
        base["font_size_seq"] = 13
        base["font_size_label"] = 11
        base["legend"] = False
        base["connectors"] = True
        base["connector_width"] = 1.1
        base["connector_alpha"] = 0.78
        base["connector_dash"] = ()
        return base

    base = _base_sequence_style_overrides()
    if view_id in {"payload", "split_payload", "assembled_payload"}:
        base["legend"] = False
    if view_id == "assembled_payload":
        base["padding_y"] = 28.0
    return base


__all__ = [
    "build_payload_view_title",
    "build_yiu_style_overrides",
]
