from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dnadesign.densegen.src.adapters.sources import PWMMemeDataSource

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
