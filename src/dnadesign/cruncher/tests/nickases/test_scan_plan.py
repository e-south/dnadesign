"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/nickases/test_scan_plan.py

Tests for precomputed nickase motif scan plans.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.nickases.models import NickaseCatalogEntry
from dnadesign.cruncher.nickases.scan_plan import build_entry_scan_plans, window_matches_plan


def test_build_entry_scan_plans_returns_forward_and_reverse_for_non_palindromic_motif() -> None:
    entry = NickaseCatalogEntry(
        id="Nt.Bpu10I",
        specificity_id="Bpu10I",
        motif_top_5to3="CCTNAGC",
        top_cut_offset=2,
    )

    plans = build_entry_scan_plans(entry)

    assert [plan.orientation for plan in plans] == ["forward", "reverse"]
    assert plans[0].motif_text == "CCTNAGC"
    assert plans[1].motif_text == "GCTNAGG"


def test_build_entry_scan_plans_deduplicates_palindromic_motif() -> None:
    entry = NickaseCatalogEntry(
        id="Nt.Pal",
        specificity_id="Pal",
        motif_top_5to3="ACGT",
        top_cut_offset=1,
    )

    plans = build_entry_scan_plans(entry)

    assert len(plans) == 1
    assert plans[0].orientation == "forward"
    assert plans[0].motif_text == "ACGT"


def test_window_matches_plan_honors_iupac_ambiguity_without_normalization_roundtrip() -> None:
    entry = NickaseCatalogEntry(
        id="Nt.Bpu10I",
        specificity_id="Bpu10I",
        motif_top_5to3="CCTNAGC",
        top_cut_offset=2,
    )
    forward_plan, reverse_plan = build_entry_scan_plans(entry)

    assert window_matches_plan("AACCTCAGCTT", start=2, plan=forward_plan) is True
    assert window_matches_plan("AAGCTTAGGTT", start=2, plan=reverse_plan) is True
    assert window_matches_plan("AACCGCAGCTT", start=2, plan=forward_plan) is False
