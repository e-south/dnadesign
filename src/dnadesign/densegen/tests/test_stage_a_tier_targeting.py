"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_stage_a_tier_targeting.py

Tier targeting helpers for Stage-A PWM sampling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.densegen.src.adapters.sources.stage_a_pipeline import (
    _evaluate_tier_target,
    _score_norm_by_tier,
    _score_norm_denominator_by_seq,
)
from dnadesign.densegen.src.adapters.sources.stage_a_sampling_utils import (
    _pwm_theoretical_max_score,
    build_log_odds,
    select_pwm_window_by_length,
)


def test_tier_target_requirement_computation() -> None:
    required, met = _evaluate_tier_target(
        n_sites=200,
        target_tier_fraction=0.001,
        eligible_unique=150_000,
    )
    assert required == 200_000
    assert met is False


def test_score_norm_by_tier_summary() -> None:
    scores_by_seq = {"a": 10.0, "b": 8.0, "c": 6.0, "d": 4.0}
    tier_by_seq = {"a": 0, "b": 1, "c": 2, "d": 3}
    summary = _score_norm_by_tier(scores_by_seq, tier_by_seq, denominator=10.0)
    assert summary is not None
    assert summary["tier0"] == {"min": 1.0, "median": 1.0, "max": 1.0}
    assert summary["tier1"] == {"min": 0.8, "median": 0.8, "max": 0.8}
    assert summary["tier2"] == {"min": 0.6, "median": 0.6, "max": 0.6}
    assert summary["rest"] == {"min": 0.4, "median": 0.4, "max": 0.4}


def test_score_norm_denominator_by_seq_uses_window_max() -> None:
    matrix = [
        {"A": 0.97, "C": 0.01, "G": 0.01, "T": 0.01},
        {"A": 0.97, "C": 0.01, "G": 0.01, "T": 0.01},
        {"A": 0.97, "C": 0.01, "G": 0.01, "T": 0.01},
        {"A": 0.97, "C": 0.01, "G": 0.01, "T": 0.01},
        {"A": 0.97, "C": 0.01, "G": 0.01, "T": 0.01},
    ]
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    sequences = ["AAAAA", "AAAA"]
    denom_by_seq = _score_norm_denominator_by_seq(
        sequences,
        matrix=matrix,
        background=background,
    )
    log_odds = build_log_odds(matrix, background, smoothing_alpha=0.0)
    full_max = _pwm_theoretical_max_score(log_odds)
    window = select_pwm_window_by_length(matrix=matrix, log_odds=log_odds, length=4)
    window_max = _pwm_theoretical_max_score(window.log_odds)
    assert denom_by_seq["AAAAA"] == pytest.approx(full_max)
    assert denom_by_seq["AAAA"] == pytest.approx(window_max)
