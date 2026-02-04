"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_postprocess_forbidden_kmer.py

Unit tests for postprocess forbidden-kmer validation helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.densegen.src.core.pipeline.sequence_validation import _find_forbidden_kmer, _promoter_windows


def test_postprocess_forbidden_kmer_detects_outside_window() -> None:
    seq = "TTGACAGGTATAATTTGACA"
    fixed_elements = {
        "promoter_constraints": [
            {
                "upstream": "TTGACA",
                "downstream": "TATAAT",
                "spacer_length": [2, 2],
            }
        ]
    }
    windows = _promoter_windows(seq, fixed_elements)
    assert windows == [(0, 6), (8, 14)]
    hit = _find_forbidden_kmer(seq, ["TTGACA", "TATAAT"], windows)
    assert hit == ("TTGACA", 14)
