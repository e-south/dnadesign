"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/nickases/test_selection.py

Contract tests for nickase selection ordering used by snapback solve.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.nickases.models import NickaseCatalogEntry, NickaseSelectionProfile
from dnadesign.cruncher.nickases.selection import snapback_entry_priority_key


def _entry(
    entry_id: str,
    *,
    outside_site: bool,
    snapback_tier: str,
) -> NickaseCatalogEntry:
    return NickaseCatalogEntry(
        id=entry_id,
        specificity_id=entry_id,
        motif_top_5to3="ATGC",
        top_cut_offset=1,
        selection=NickaseSelectionProfile(
            outside_site=outside_site,
            snapback_tier=snapback_tier,  # type: ignore[arg-type]
            commercial_confidence="primary_vendor_current",
        ),
    )


def test_snapback_entry_priority_prefers_outside_site_within_same_tier() -> None:
    outside_site = _entry("outside", outside_site=True, snapback_tier="tier1")
    inside_site = _entry("inside", outside_site=False, snapback_tier="tier1")

    assert snapback_entry_priority_key(outside_site) < snapback_entry_priority_key(inside_site)


def test_snapback_entry_priority_prefers_tier_before_outside_site() -> None:
    stronger_tier_inside = _entry("stronger", outside_site=False, snapback_tier="tier1")
    weaker_tier_outside = _entry("weaker", outside_site=True, snapback_tier="tier2")

    assert snapback_entry_priority_key(stronger_tier_inside) < snapback_entry_priority_key(weaker_tier_outside)
