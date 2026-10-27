"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/plotting/test_plot_style_helpers.py

Tests for plot style helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.densegen.src.utils.plot_style import format_regulator_label
from dnadesign.densegen.src.viz.plotting import _palette


def test_format_regulator_label_splits_consensus() -> None:
    assert format_regulator_label("lexA_CTGTATAWAWWHACA") == "lexA\nCTGTATAWAWWHACA"


def test_format_regulator_label_wraps_long_ids() -> None:
    wrapped = format_regulator_label("very_long_label_with_many_parts", wrap_width=8)
    assert "\n" in wrapped


def test_palette_colorblind2_alias() -> None:
    expected = _palette({"palette": "okabe_ito"}, 5)
    observed = _palette({"palette": "colorblind2"}, 5)
    assert observed == expected
