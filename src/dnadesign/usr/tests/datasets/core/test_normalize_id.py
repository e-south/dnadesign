"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/tests/datasets/core/test_normalize_id.py

Tests deterministic ID hashing rules and delimiter handling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import hashlib

from dnadesign.usr.src.contracts import ID_DELIMITER, compute_id


def test_compute_id_uses_delimiter_and_utf8() -> None:
    expected = hashlib.sha1(f"dna{ID_DELIMITER}ACGT".encode("utf-8")).hexdigest()
    assert compute_id("dna", "ACGT") == expected
