from __future__ import annotations

from pathlib import Path

import numpy as np

from dnadesign.densegen.src.adapters.sources import PWMJasparDataSource, PWMMatrixCSVDataSource

JASPAR_TEXT = """>M1
A [ 3 1 0 ]
C [ 0 3 1 ]
G [ 1 0 3 ]
T [ 0 1 0 ]
"""


def test_pwm_jaspar_sampling(tmp_path: Path) -> None:
    jaspar_path = tmp_path / "motifs.pfm"
    jaspar_path.write_text(JASPAR_TEXT)
    ds = PWMJasparDataSource(
        path=str(jaspar_path),
        cfg_path=tmp_path / "config.yaml",
        input_name="demo_input",
        motif_ids=["M1"],
        sampling={
            "strategy": "stochastic",
            "n_sites": 4,
            "oversample_factor": 3,
            "score_threshold": -10.0,
            "score_percentile": None,
        },
    )
    entries, df = ds.load_data(rng=np.random.default_rng(0))
    assert len(entries) == 4
    assert set(df["tf"].tolist()) == {"M1"}


def test_pwm_matrix_csv_sampling(tmp_path: Path) -> None:
    csv_path = tmp_path / "matrix.csv"
    csv_path.write_text("A,C,G,T\n3,0,1,0\n1,3,0,1\n0,1,3,0\n")
    ds = PWMMatrixCSVDataSource(
        path=str(csv_path),
        cfg_path=tmp_path / "config.yaml",
        input_name="demo_input",
        motif_id="M1",
        columns={"A": "A", "C": "C", "G": "G", "T": "T"},
        sampling={
            "strategy": "consensus",
            "n_sites": 1,
            "oversample_factor": 2,
            "score_threshold": -10.0,
            "score_percentile": None,
        },
    )
    entries, df = ds.load_data(rng=np.random.default_rng(1))
    assert len(entries) == 1
    assert df["tf"].unique().tolist() == ["M1"]
