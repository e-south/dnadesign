"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/nickases/test_scanning.py

Tests for shared nickase scanning helpers.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.nickases.models import NickaseCatalogEntry
from dnadesign.cruncher.nickases.scanning import (
    enumerate_site_instances,
    enumerate_site_instances_starting_at_or_after,
    suffix_sensitive_scan_start,
)


def _entry() -> NickaseCatalogEntry:
    return NickaseCatalogEntry(
        id="Nt.Bpu10I",
        specificity_id="Bpu10I",
        motif_top_5to3="CCTNAGC",
        top_cut_offset=2,
    )


def test_range_scan_matches_filtered_full_scan() -> None:
    sequence = "CCTCAGCCCTTAGCAGGCCTAAGC"
    entry = _entry()

    full = enumerate_site_instances(sequence, coordinate_offset=0, entry=entry)
    filtered = enumerate_site_instances_starting_at_or_after(
        sequence,
        coordinate_offset=0,
        entry=entry,
        start_min=8,
    )

    assert [match.key() for match in filtered] == [match.key() for match in full if match.site.start >= 8]


def test_range_scan_returns_empty_when_start_is_past_last_window() -> None:
    entry = _entry()

    matches = enumerate_site_instances_starting_at_or_after(
        "CCTCAGCCCTTAGCAGGCCTAAGC",
        coordinate_offset=0,
        entry=entry,
        start_min=100,
    )

    assert matches == []


def test_suffix_sensitive_scan_start_captures_only_windows_that_can_touch_appended_suffix() -> None:
    entry = _entry()

    assert suffix_sensitive_scan_start(entry, prefix_length=8) == 2
    assert suffix_sensitive_scan_start(entry, prefix_length=7) == 1
    assert suffix_sensitive_scan_start(entry, prefix_length=3) == 0


def test_prefix_matches_plus_suffix_sensitive_scan_matches_full_scan_for_appended_sequence() -> None:
    entry = _entry()
    input_sequence = "AACCTCAG"
    designed_sequence = f"{input_sequence}CTT"
    prefix_matches = enumerate_site_instances(input_sequence, coordinate_offset=0, entry=entry)
    suffix_matches = enumerate_site_instances_starting_at_or_after(
        designed_sequence,
        coordinate_offset=0,
        entry=entry,
        start_min=suffix_sensitive_scan_start(entry, prefix_length=len(input_sequence)),
    )
    full = enumerate_site_instances(designed_sequence, coordinate_offset=0, entry=entry)

    assert [match.key() for match in [*prefix_matches, *suffix_matches]] == [match.key() for match in full]
