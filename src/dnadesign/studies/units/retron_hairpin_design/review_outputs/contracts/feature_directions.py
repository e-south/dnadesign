"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/contracts/feature_directions.py

Feature-direction contract for Retron MSD Benchling GenBank handoff records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

FEATURE_DIRECTION_BY_ROLE = {
    "flank_3p": "forward",
    "stem_base_right": "forward",
    "payload_complement": "forward",
    "snapback_foldback_return": "forward",
    "foldback_return": "forward",
    "flank_5p": "reverse",
    "stem_base_left": "reverse",
    "payload_primary": "reverse",
    "snapback_retained_stem": "reverse",
    "foldback_stem": "reverse",
    "snapback_cap": "undirected",
    "snapback_foldback_geometry": "undirected",
    "foldback": "undirected",
}


def feature_direction_for_role(role: str) -> str | None:
    return FEATURE_DIRECTION_BY_ROLE.get(role)


__all__ = ["FEATURE_DIRECTION_BY_ROLE", "feature_direction_for_role"]
