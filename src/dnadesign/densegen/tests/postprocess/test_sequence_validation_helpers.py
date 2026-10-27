"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/postprocess/test_sequence_validation_helpers.py

Unit tests for postprocess sequence validation helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.densegen.src.core.pipeline.sequence_validation import _apply_pad_offsets


def test_apply_pad_offsets_shifts_coordinates_for_5prime_padding() -> None:
    used_tfbs_detail = [
        {"tf": "lexA", "tfbs": "ACGT", "offset": 2, "length": 4},
        {"tf": "cpxR", "tfbs": "TTAA", "offset_raw": 10, "length": 4},
    ]
    shifted = _apply_pad_offsets(
        used_tfbs_detail=used_tfbs_detail,
        pad_meta={"used": True, "end": "5prime", "bases": 3},
    )
    assert shifted[0]["offset_raw"] == 2
    assert shifted[0]["pad_left"] == 3
    assert shifted[0]["offset"] == 5
    assert shifted[0]["end"] == 9
    assert shifted[1]["offset_raw"] == 10
    assert shifted[1]["pad_left"] == 3
    assert shifted[1]["offset"] == 13
    assert shifted[1]["end"] == 17
