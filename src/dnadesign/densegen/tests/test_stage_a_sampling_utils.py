"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_stage_a_sampling_utils.py

Stage-A PWM sampling utility tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math

import pytest

from dnadesign.densegen.src.adapters.sources import stage_a_sampling_utils


def test_pwm_theoretical_max_score_matches_log_odds_maxima() -> None:
    matrix = [
        {"A": 0.8, "C": 0.1, "G": 0.05, "T": 0.05},
        {"A": 0.1, "C": 0.1, "G": 0.7, "T": 0.1},
    ]
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    log_odds = stage_a_sampling_utils.build_log_odds(matrix, background, smoothing_alpha=0.0)
    expected = math.log(0.8 / 0.25) + math.log(0.7 / 0.25)
    theoretical = stage_a_sampling_utils._pwm_theoretical_max_score(log_odds)
    assert theoretical == pytest.approx(expected, rel=1e-6)
