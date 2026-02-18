from __future__ import annotations

import numpy as np
import pytest

from dnadesign.densegen.src.adapters.sources.pwm_sampling import sample_pwm_sites
from dnadesign.densegen.src.core.stage_a.stage_a_sampling_utils import select_pwm_window_by_length
from dnadesign.densegen.src.core.stage_a.stage_a_types import PWMMotif
from dnadesign.densegen.src.integrations.meme_suite import resolve_executable
from dnadesign.densegen.tests.pwm_sampling_fixtures import (
    fixed_candidates_mining,
    selection_mmr,
    selection_top_score,
)

pytestmark = pytest.mark.fimo

_FIMO_MISSING = resolve_executable("fimo", tool_path=None) is None


def test_pwm_sampling_trim_window_selects_max_info() -> None:
    motif = PWMMotif(
        motif_id="trim",
        matrix=[
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 1.0, "C": 0.0, "G": 0.0, "T": 0.0},
            {"A": 1.0, "C": 0.0, "G": 0.0, "T": 0.0},
        ],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )
    rng = np.random.default_rng(0)
    if _FIMO_MISSING:
        pytest.skip("fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).")
    seqs = sample_pwm_sites(
        rng,
        motif,
        strategy="consensus",
        n_sites=1,
        mining=fixed_candidates_mining(batch_size=1, candidates=1),
        selection=selection_top_score(),
        length_policy="exact",
        length_range=None,
        trim_window_length=2,
        trim_window_strategy="max_info",
    )
    assert seqs == ["AA"]


def test_pwm_sampling_trim_length_rejects_too_long() -> None:
    motif = PWMMotif(
        motif_id="trim",
        matrix=[
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        ],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="exceeds motif width"):
        sample_pwm_sites(
            rng,
            motif,
            strategy="consensus",
            n_sites=1,
            mining=fixed_candidates_mining(batch_size=1, candidates=1),
            selection=selection_top_score(),
            length_policy="exact",
            length_range=None,
            trim_window_length=3,
            trim_window_strategy="max_info",
        )


def test_pwm_sampling_range_trims_long_motif_to_max_info_window() -> None:
    motif = PWMMotif(
        motif_id="range_trim",
        matrix=[
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 1.0, "C": 0.0, "G": 0.0, "T": 0.0},
            {"A": 1.0, "C": 0.0, "G": 0.0, "T": 0.0},
            {"A": 1.0, "C": 0.0, "G": 0.0, "T": 0.0},
        ],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )
    rng = np.random.default_rng(0)
    if _FIMO_MISSING:
        pytest.skip("fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).")
    seqs = sample_pwm_sites(
        rng,
        motif,
        strategy="consensus",
        n_sites=1,
        mining=fixed_candidates_mining(batch_size=1, candidates=1),
        selection=selection_top_score(),
        length_policy="range",
        length_range=[3, 3],
    )
    assert seqs == ["AAA"]


def test_pwm_sampling_mmr_range_requires_uniform_core_length() -> None:
    motif = PWMMotif(
        motif_id="range_mmr",
        matrix=[
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
            {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        ],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="MMR requires a fixed trim window"):
        sample_pwm_sites(
            rng,
            motif,
            strategy="consensus",
            n_sites=1,
            mining=fixed_candidates_mining(batch_size=1, candidates=1),
            selection=selection_mmr(alpha=0.5),
            length_policy="range",
            length_range=[3, 5],
        )


def test_select_pwm_window_by_length_prefers_max_info() -> None:
    matrix = [
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
        {"A": 1.0, "C": 0.0, "G": 0.0, "T": 0.0},
        {"A": 1.0, "C": 0.0, "G": 0.0, "T": 0.0},
        {"A": 1.0, "C": 0.0, "G": 0.0, "T": 0.0},
    ]
    log_odds = [
        {"A": 0.0, "C": 0.0, "G": 0.0, "T": 0.0},
        {"A": 0.0, "C": 0.0, "G": 0.0, "T": 0.0},
        {"A": 2.0, "C": -1.0, "G": -1.0, "T": -1.0},
        {"A": 2.0, "C": -1.0, "G": -1.0, "T": -1.0},
        {"A": 2.0, "C": -1.0, "G": -1.0, "T": -1.0},
    ]
    windowed = select_pwm_window_by_length(matrix=matrix, log_odds=log_odds, length=3)
    assert windowed.start == 2
    assert len(windowed.matrix) == 3
