"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/nickases/selection.py

Central nickase selection helpers for downstream workflow ranking/reporting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable

from dnadesign.cruncher.nickases.models import NickaseCatalogEntry

_SNAPBACK_TIER_RANK = {
    "tier1": 0,
    "tier2": 1,
    "tier3": 2,
    None: 3,
}
_COMMERCIAL_CONFIDENCE_RANK = {
    "primary_vendor_current": 0,
    "secondary_vendor_current": 1,
    "produced_on_demand": 2,
    "literature_only": 3,
    None: 4,
}


def snapback_entry_priority_key(entry: NickaseCatalogEntry) -> tuple[object, ...]:
    selection = entry.selection
    warning_codes = selection.warning_codes if selection is not None else []
    return (
        _SNAPBACK_TIER_RANK[selection.snapback_tier if selection is not None else None],
        0 if selection is not None and selection.outside_site is True else 1 if selection is not None else 2,
        -(entry.motif_len or len(entry.motif_top_5to3)),
        _COMMERCIAL_CONFIDENCE_RANK[selection.commercial_confidence if selection is not None else None],
        len(warning_codes),
        entry.id,
    )


def matching_nickase_warning_codes(
    entry: NickaseCatalogEntry,
    *,
    warning_codes: Iterable[str],
) -> list[str]:
    selection = entry.selection
    if selection is None or not selection.warning_codes:
        return []
    present = set(selection.warning_codes)
    return [code for code in warning_codes if code in present]


__all__ = ["matching_nickase_warning_codes", "snapback_entry_priority_key"]
