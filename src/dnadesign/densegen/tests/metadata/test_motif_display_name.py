"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/metadata/test_motif_display_name.py

Unit tests for motif display name formatting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.densegen.src.core.motif_labels import motif_display_name


def test_motif_display_name_prefers_tf_name() -> None:
    assert motif_display_name("lexA_CTGT", "LexA") == "LexA"


def test_motif_display_name_uses_prefix() -> None:
    assert motif_display_name("lexA_CTGT", None) == "lexA"


def test_motif_display_name_returns_raw_when_no_prefix() -> None:
    assert motif_display_name("LexA", None) == "LexA"
