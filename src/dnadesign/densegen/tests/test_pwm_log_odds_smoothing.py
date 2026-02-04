"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/tests/test_pwm_log_odds_smoothing.py

Checks PWM log-odds smoothing and sampling stability.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.densegen.src.adapters.sources.pwm_sampling import sample_pwm_sites
from dnadesign.densegen.src.adapters.sources.stage_a.stage_a_sampling_utils import (
    _pwm_consensus,
    _pwm_consensus_iupac,
    _pwm_theoretical_max_score,
    build_log_odds,
    parse_bgfile,
    score_sequence,
)
from dnadesign.densegen.src.adapters.sources.stage_a.stage_a_types import PWMMotif
from dnadesign.densegen.src.integrations.meme_suite import resolve_executable
from dnadesign.densegen.tests.pwm_sampling_fixtures import fixed_candidates_mining, selection_top_score

_FIMO_MISSING = resolve_executable("fimo", tool_path=None) is None


def test_pwm_log_odds_smoothing_finite() -> None:
    matrix = [
        {"A": 1.0, "C": 0.0, "G": 0.0, "T": 0.0},
        {"A": 0.0, "C": 1.0, "G": 0.0, "T": 0.0},
        {"A": 0.0, "C": 0.0, "G": 1.0, "T": 0.0},
    ]
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    log_odds = build_log_odds(matrix, background)
    assert all(np.isfinite(val) for row in log_odds for val in row.values())

    if _FIMO_MISSING:
        pytest.skip("fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).")

    motif = PWMMotif(motif_id="M1", matrix=matrix, background=background)
    rng = np.random.default_rng(0)
    sites = sample_pwm_sites(
        rng,
        motif,
        strategy="stochastic",
        n_sites=1,
        mining=fixed_candidates_mining(batch_size=5, candidates=5, log_every_batches=1),
        selection=selection_top_score(),
    )
    assert len(sites) == 1
    core = sites[0][: len(matrix)]
    score = score_sequence(core, matrix, background=background)
    assert np.isfinite(score)


def test_pwm_sampling_theoretical_max_uses_matrix_log_odds() -> None:
    if _FIMO_MISSING:
        pytest.skip("fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).")

    matrix = [
        {"A": 0.8, "C": 0.1, "G": 0.05, "T": 0.05},
        {"A": 0.1, "C": 0.8, "G": 0.05, "T": 0.05},
    ]
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    log_odds = build_log_odds(matrix, background, smoothing_alpha=0.0)
    mismatched = [{base: val * 0.5 for base, val in row.items()} for row in log_odds]
    motif = PWMMotif(motif_id="M1", matrix=matrix, background=background, log_odds=mismatched)

    expected = _pwm_theoretical_max_score(log_odds)
    rng = np.random.default_rng(1)
    _selected, summary = sample_pwm_sites(
        rng,
        motif,
        strategy="consensus",
        n_sites=1,
        mining=fixed_candidates_mining(batch_size=1, candidates=1, log_every_batches=1),
        selection=selection_top_score(),
        return_summary=True,
    )

    assert summary is not None
    assert summary.pwm_theoretical_max_score == pytest.approx(expected, rel=1e-6)


def test_pwm_sampling_uses_bgfile_for_theoretical_max(tmp_path) -> None:
    if _FIMO_MISSING:
        pytest.skip("fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).")

    matrix = [
        {"A": 0.9, "C": 0.05, "G": 0.03, "T": 0.02},
        {"A": 0.1, "C": 0.1, "G": 0.7, "T": 0.1},
    ]
    motif_background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    bgfile = tmp_path / "bg.txt"
    bgfile.write_text("A 0.7\nC 0.1\nG 0.1\nT 0.1\n")
    bg = parse_bgfile(bgfile)
    log_odds = build_log_odds(matrix, bg, smoothing_alpha=0.0)
    expected_max = _pwm_theoretical_max_score(log_odds)
    expected_consensus = score_sequence(_pwm_consensus(matrix), matrix, background=bg)

    motif = PWMMotif(motif_id="M_bg", matrix=matrix, background=motif_background)
    rng = np.random.default_rng(2)
    _selected, summary = sample_pwm_sites(
        rng,
        motif,
        strategy="consensus",
        n_sites=1,
        mining=fixed_candidates_mining(batch_size=1, candidates=1, log_every_batches=1),
        selection=selection_top_score(),
        return_summary=True,
        bgfile=bgfile,
    )

    assert summary is not None
    assert summary.pwm_theoretical_max_score == pytest.approx(expected_max, rel=1e-6)
    assert summary.pwm_consensus_score == pytest.approx(expected_consensus, rel=1e-6)


def test_pwm_iupac_consensus_from_pwm() -> None:
    matrix = [
        {"A": 0.7, "C": 0.1, "G": 0.1, "T": 0.1},
        {"A": 0.4, "C": 0.4, "G": 0.1, "T": 0.1},
        {"A": 0.34, "C": 0.33, "G": 0.01, "T": 0.32},
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    ]
    assert _pwm_consensus_iupac(matrix) == "AMHN"


def test_pwm_iupac_consensus_includes_secondary_bases() -> None:
    matrix = [
        {"A": 0.6, "C": 0.3, "G": 0.1, "T": 0.0},
    ]
    assert _pwm_consensus_iupac(matrix) == "M"
