"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/scar_nick/semantics.py

Shared scar_nick ontology helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal

PROFILE_ORDER_S3S2S1S0 = "S3_S2_S1_S0"
S_SITE_ORDER = ("S3", "S2", "S1", "S0")

ScarNickStrand = Literal["top", "bottom"]
ScarNickVisualRow = Literal["primary", "complement"]


def surviving_strand_for_nick(nicked_strand: ScarNickStrand | None) -> ScarNickStrand | None:
    if nicked_strand is None:
        return None
    if nicked_strand == "top":
        return "bottom"
    if nicked_strand == "bottom":
        return "top"
    raise ValueError(f"unknown scar_nick strand: {nicked_strand!r}")


def row_for_strand(strand: ScarNickStrand) -> ScarNickVisualRow:
    if strand == "top":
        return "primary"
    if strand == "bottom":
        return "complement"
    raise ValueError(f"unknown scar_nick strand: {strand!r}")


__all__ = [
    "PROFILE_ORDER_S3S2S1S0",
    "S_SITE_ORDER",
    "ScarNickStrand",
    "ScarNickVisualRow",
    "row_for_strand",
    "surviving_strand_for_nick",
]
