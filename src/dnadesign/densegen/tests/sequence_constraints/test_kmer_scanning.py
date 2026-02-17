"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/sequence_constraints/test_kmer_scanning.py

Unit tests for strand-aware kmer scanning and coordinate mapping.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.densegen.src.core.sequence_constraints.kmers import find_kmer_matches, reverse_complement


def test_reverse_complement_basic() -> None:
    assert reverse_complement("TTGACA") == "TGTCAA"


def test_find_kmer_matches_detects_forward_and_reverse_with_forward_coordinates() -> None:
    seq = "AAATGTCAAGG"
    matches = find_kmer_matches(sequence=seq, pattern="TTGACA", strands="both")
    assert matches == [
        {
            "pattern": "TTGACA",
            "strand": "-",
            "position": 3,
            "matched_seq": "TGTCAA",
        }
    ]


def test_find_kmer_matches_detects_join_spanning_occurrences() -> None:
    left = "AAAAATTGAC"
    right = "ATTT"
    seq = left + right
    matches = find_kmer_matches(sequence=seq, pattern="TTGACA", strands="forward")
    assert matches == [
        {
            "pattern": "TTGACA",
            "strand": "+",
            "position": 5,
            "matched_seq": "TTGACA",
        }
    ]
