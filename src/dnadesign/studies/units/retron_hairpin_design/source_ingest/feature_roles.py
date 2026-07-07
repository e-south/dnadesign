"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/source_ingest/feature_roles.py

Feature-role normalization for MSD region GenBank records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from typing import Sequence

_LABEL_ROLE_ALIASES = {
    "3 flanking": "flank_3p",
    "3' flanking": "flank_3p",
    "5 flanking": "flank_5p",
    "5' flanking": "flank_5p",
    "right base": "stem_base_right",
    "left base": "stem_base_left",
    "foldback return": "snapback_foldback_return",
    "foldback": "snapback_foldback_geometry",
    "wt loop": "snapback_foldback_geometry",
    "loop": "snapback_foldback_geometry",
    "cap": "snapback_cap",
    "foldback stem": "snapback_retained_stem",
}


def normalized_role_for_feature(
    *,
    labels: Sequence[str],
    source_roles: Sequence[str],
    source_start_0: int,
    source_end_0: int,
    source_length: int,
    source_strand: int | None,
) -> tuple[str | None, str | None]:
    """Return normalized role plus direct source role, if present."""

    source_role = _first_text(source_roles)
    if source_role:
        return source_role, source_role
    label = _first_text(labels)
    if not label:
        return None, None
    if source_start_0 == 0 and source_end_0 == source_length and "msd" in label.lower():
        return "msd_region", None
    norm = _normalize_label(label)
    role = _LABEL_ROLE_ALIASES.get(norm)
    if role is not None:
        if role in {"stem_base_left", "stem_base_right"} and source_end_0 - source_start_0 != 4:
            return f"{role}_annotated_span", None
        return role, None
    label_lower = label.lower()
    is_payload_label = (
        "tet operator" in label_lower or "teto" in label_lower or "tetr" in label_lower or "msd[" in label_lower
    )
    if "complement" in label_lower and is_payload_label:
        return "payload_complement", None
    if is_payload_label:
        if source_strand == 1:
            return "payload_complement", None
        if source_strand == -1:
            return "payload_primary", None
    if is_payload_label:
        return "payload_primary", None
    return None, None


def feature_label(labels: Sequence[str]) -> str:
    return _first_text(labels) or ""


def _first_text(values: Sequence[str]) -> str | None:
    for value in values:
        text = str(value).strip()
        if text:
            return text
    return None


def _normalize_label(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = text.replace("'", "'")
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return " ".join(text.split())


__all__ = ["feature_label", "normalized_role_for_feature"]
