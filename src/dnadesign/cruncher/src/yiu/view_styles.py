"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/view_styles.py

Display-title and style policy for payload-centric YIU views.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re

from dnadesign.cruncher.yiu.domain_models import NormalizedPayload
from dnadesign.cruncher.yiu.visual_system import (
    YiuViewStyleProfile,
    build_yiu_style_overrides,
    get_yiu_style_profile,
)


def _pretty_label(text: str | None) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    normalized = re.sub(r"[_-]+", " ", raw)
    return " ".join(token[:1].upper() + token[1:] for token in normalized.split())


def _motif_tf_names(normalized: NormalizedPayload) -> list[str]:
    motif_tf_names = {
        tf_name for tf_name in (str(motif.tf_name).strip() for motif in normalized.motif_context.motifs) if tf_name
    }
    return sorted(motif_tf_names)


def _payload_label(normalized: NormalizedPayload) -> str:
    return _pretty_label(normalized.payload_label) or _pretty_label(normalized.name) or "Payload"


def build_payload_view_title(normalized: NormalizedPayload) -> str:
    motif_tfs = _motif_tf_names(normalized)
    if len(motif_tfs) == 1:
        tf_label = _pretty_label(motif_tfs[0])
        motif_count = len(normalized.motif_context.motifs)
        suffix = f" ({motif_count} sites)" if motif_count > 1 else ""
        return f"{tf_label} payload{suffix}"
    return _payload_label(normalized)


__all__ = [
    "build_payload_view_title",
    "build_yiu_style_overrides",
    "YiuViewStyleProfile",
    "get_yiu_style_profile",
]
