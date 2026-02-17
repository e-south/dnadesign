"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/stage_a/test_stage_a_sampling_utils.py

Stage-A PWM sampling utility tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math

import pytest

from dnadesign.densegen.src.core.stage_a import stage_a_sampling_utils


def test_pwm_theoretical_max_score_matches_log_odds_maxima() -> None:
    matrix = [
        {"A": 0.8, "C": 0.1, "G": 0.05, "T": 0.05},
        {"A": 0.1, "C": 0.1, "G": 0.7, "T": 0.1},
    ]
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    log_odds = stage_a_sampling_utils.build_log_odds(matrix, background, smoothing_alpha=0.0)
    expected = math.log2(0.8 / 0.25) + math.log2(0.7 / 0.25)
    theoretical = stage_a_sampling_utils._pwm_theoretical_max_score(log_odds)
    assert theoretical == pytest.approx(expected, rel=1e-6)


def test_parse_bgfile_reads_background_frequencies(tmp_path) -> None:
    bgfile = tmp_path / "bg.txt"
    bgfile.write_text("# comment line\nBackground letter frequencies:\nA 0.1\nC 0.2\nG 0.3\nT 0.4\n")
    background = stage_a_sampling_utils.parse_bgfile(bgfile)
    assert background["A"] == pytest.approx(0.1, rel=1e-6)
    assert background["C"] == pytest.approx(0.2, rel=1e-6)
    assert background["G"] == pytest.approx(0.3, rel=1e-6)
    assert background["T"] == pytest.approx(0.4, rel=1e-6)


def test_parse_bgfile_rejects_inline_pairs(tmp_path) -> None:
    bgfile = tmp_path / "bg.txt"
    bgfile.write_text("A 0.7 C 0.1 G 0.1 T 0.1\n")
    with pytest.raises(ValueError, match="bgfile line must be"):
        stage_a_sampling_utils.parse_bgfile(bgfile)
