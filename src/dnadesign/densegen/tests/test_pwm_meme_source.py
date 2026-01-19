from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from dnadesign.densegen.src.adapters.sources import PWMMemeDataSource
from dnadesign.densegen.src.adapters.sources.pwm_sampling import PWMMotif, sample_pwm_sites

MEME_TEXT = """\
MEME version 4

ALPHABET= ACGT

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

MOTIF M1
letter-probability matrix: alength= 4 w= 3 nsites= 20 E= 0
0.8 0.1 0.05 0.05
0.1 0.7 0.1 0.1
0.1 0.1 0.7 0.1

MOTIF M2
letter-probability matrix: alength= 4 w= 2 nsites= 10 E= 0
0.6 0.2 0.1 0.1
0.2 0.6 0.1 0.1
"""


def test_pwm_meme_sampling_stochastic(tmp_path: Path) -> None:
    meme_path = tmp_path / "motifs.meme"
    meme_path.write_text(MEME_TEXT)
    ds = PWMMemeDataSource(
        path=str(meme_path),
        cfg_path=tmp_path / "config.yaml",
        motif_ids=["M1"],
        sampling={
            "strategy": "stochastic",
            "n_sites": 5,
            "oversample_factor": 3,
            "score_threshold": -10.0,
            "score_percentile": None,
        },
    )
    entries, df = ds.load_data(rng=np.random.default_rng(0))
    assert len(entries) == 5
    assert df is not None
    assert set(df["tf"].tolist()) == {"M1"}


def test_pwm_meme_consensus_requires_one_site(tmp_path: Path) -> None:
    meme_path = tmp_path / "motifs.meme"
    meme_path.write_text(MEME_TEXT)
    ds = PWMMemeDataSource(
        path=str(meme_path),
        cfg_path=tmp_path / "config.yaml",
        motif_ids=["M2"],
        sampling={
            "strategy": "consensus",
            "n_sites": 2,
            "oversample_factor": 2,
            "score_threshold": -10.0,
            "score_percentile": None,
        },
    )
    with pytest.raises(ValueError, match="consensus"):
        ds.load_data(rng=np.random.default_rng(1))


def test_pwm_sampling_cap_warns(caplog: pytest.LogCaptureFixture) -> None:
    motif = PWMMotif(
        motif_id="M1",
        matrix=[
            {"A": 0.7, "C": 0.1, "G": 0.1, "T": 0.1},
            {"A": 0.1, "C": 0.7, "G": 0.1, "T": 0.1},
            {"A": 0.1, "C": 0.1, "G": 0.7, "T": 0.1},
        ],
        background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
    )
    rng = np.random.default_rng(0)
    with caplog.at_level(logging.WARNING):
        sample_pwm_sites(
            rng,
            motif,
            strategy="stochastic",
            n_sites=10,
            oversample_factor=10,
            max_candidates=20,
            score_threshold=-10.0,
            score_percentile=None,
        )
    assert "capped candidate generation" in caplog.text


def test_pwm_sampling_error_context(tmp_path: Path) -> None:
    meme_path = tmp_path / "motifs.meme"
    meme_path.write_text(MEME_TEXT)
    ds = PWMMemeDataSource(
        path=str(meme_path),
        cfg_path=tmp_path / "config.yaml",
        motif_ids=["M1"],
        sampling={
            "strategy": "stochastic",
            "n_sites": 5,
            "oversample_factor": 2,
            "max_candidates": 4,
            "score_threshold": 1000.0,
            "score_percentile": None,
        },
    )
    with pytest.raises(ValueError) as exc:
        ds.load_data(rng=np.random.default_rng(2))
    msg = str(exc.value)
    assert "motif 'M1'" in msg
    assert "width=" in msg
    assert "requested" in msg
    assert "unique candidates" in msg or "Unique candidates" in msg
