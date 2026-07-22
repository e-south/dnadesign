"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/release_enzymes/selection.py

Deterministic release-enzyme priority helpers for released-product snapback.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.release_enzymes.models import ReleaseEnzymeEntry

_COMMERCIAL_CONFIDENCE_RANK = {
    "primary_vendor_current": 0,
    "secondary_vendor_current": 1,
    "legacy_vendor_page": 2,
}


def release_entry_priority_key(entry: ReleaseEnzymeEntry) -> tuple[object, ...]:
    return (
        _COMMERCIAL_CONFIDENCE_RANK[entry.commercial_confidence],
        len(entry.warning_codes),
        entry.proximal_reach_from_site_end,
        entry.recognition_len,
        entry.variant_id,
    )


__all__ = ["release_entry_priority_key"]
