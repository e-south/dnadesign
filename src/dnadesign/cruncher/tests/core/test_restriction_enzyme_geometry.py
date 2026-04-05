"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_restriction_enzyme_geometry.py

Regression tests for shared restriction-enzyme cut geometry.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.bio import derive_cut_geometry


def test_bsmbi_forward_and_reverse_sites_swap_cut_offsets_across_strands() -> None:
    forward = derive_cut_geometry(
        "CGTCTCAAACCGG",
        start=0,
        recognition_sequence="CGTCTC",
        orientation="forward",
        top_cut_offset=7,
        bottom_cut_offset=11,
    )
    reverse = derive_cut_geometry(
        "AACCGGTGAGACG",
        start=7,
        recognition_sequence="CGTCTC",
        orientation="reverse",
        top_cut_offset=7,
        bottom_cut_offset=11,
    )

    assert forward.top_boundary == 7
    assert forward.bottom_boundary == 11
    assert forward.overhang_sequence == "AACC"
    assert forward.protruding_strand == "primary"

    assert reverse.top_boundary == 2
    assert reverse.bottom_boundary == 6
    assert reverse.overhang_sequence == "CCGG"
    assert reverse.protruding_strand == "primary"
