"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/tests/test_pwm_meme_source.py

Stage-A PWM sampling via MEME sources.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from dnadesign.densegen.src.adapters.sources import PWMMemeDataSource
from dnadesign.densegen.src.adapters.sources.pwm_sampling import sample_pwm_sites
from dnadesign.densegen.src.adapters.sources.stage_a.stage_a_types import PWMMotif
from dnadesign.densegen.src.integrations.meme_suite import resolve_executable
from dnadesign.densegen.tests.pwm_sampling_fixtures import (
    fixed_candidates_mining,
    sampling_config,
    selection_top_score,
)

_FIMO_MISSING = resolve_executable("fimo", tool_path=None) is None

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


@pytest.mark.skipif(
    _FIMO_MISSING,
    reason="fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).",
)
def test_pwm_meme_sampling_stochastic(tmp_path: Path) -> None:
    meme_path = tmp_path / "motifs.meme"
    meme_path.write_text(MEME_TEXT)
    ds = PWMMemeDataSource(
        path=str(meme_path),
        cfg_path=tmp_path / "config.yaml",
        input_name="demo_input",
        motif_ids=["M1"],
        sampling=sampling_config(
            n_sites=5,
            strategy="stochastic",
            mining=fixed_candidates_mining(batch_size=10, candidates=50),
        ),
    )
    entries, df, _summaries = ds.load_data(rng=np.random.default_rng(0))
    assert len(entries) == 5
    assert df is not None
    assert set(df["tf"].tolist()) == {"M1"}


def test_pwm_meme_consensus_requires_one_site(tmp_path: Path) -> None:
    meme_path = tmp_path / "motifs.meme"
    meme_path.write_text(MEME_TEXT)
    with pytest.raises(ValueError, match="consensus"):
        sampling_config(
            n_sites=2,
            strategy="consensus",
            mining=fixed_candidates_mining(batch_size=10, candidates=10),
        )


def test_pwm_sampling_shortfall_warns_on_cap(caplog: pytest.LogCaptureFixture) -> None:
    if _FIMO_MISSING:
        pytest.skip("fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).")
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
            mining=fixed_candidates_mining(batch_size=5, candidates=5, log_every_batches=1),
            selection=selection_top_score(),
        )
    assert "shortfall" in caplog.text.lower()


def test_pwm_sampling_shortfall_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    if _FIMO_MISSING:
        pytest.skip("fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).")
    meme_path = tmp_path / "motifs.meme"
    meme_path.write_text(MEME_TEXT)
    ds = PWMMemeDataSource(
        path=str(meme_path),
        cfg_path=tmp_path / "config.yaml",
        input_name="demo_input",
        motif_ids=["M1"],
        sampling=sampling_config(
            n_sites=5,
            strategy="stochastic",
            mining=fixed_candidates_mining(batch_size=2, candidates=4, log_every_batches=1),
        ),
    )
    with caplog.at_level(logging.WARNING):
        ds.load_data(rng=np.random.default_rng(2))
    assert "shortfall" in caplog.text
