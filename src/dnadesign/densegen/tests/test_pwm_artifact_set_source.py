from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dnadesign.densegen.src.adapters.sources import PWMArtifactSetDataSource
from dnadesign.densegen.src.adapters.sources.pwm_sampling import build_log_odds


def _write_artifact(path: Path, motif_id: str) -> None:
    matrix = [
        {"A": 0.7, "C": 0.1, "G": 0.1, "T": 0.1},
        {"A": 0.1, "C": 0.7, "G": 0.1, "T": 0.1},
        {"A": 0.1, "C": 0.1, "G": 0.7, "T": 0.1},
    ]
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    log_odds = build_log_odds(matrix, background)
    artifact = {
        "schema_version": "1.0",
        "producer": "test",
        "motif_id": motif_id,
        "alphabet": "ACGT",
        "matrix_semantics": "probabilities",
        "background": background,
        "probabilities": matrix,
        "log_odds": log_odds,
    }
    path.write_text(json.dumps(artifact))


def test_pwm_artifact_set_sampling(tmp_path: Path) -> None:
    a_path = tmp_path / "m1.json"
    b_path = tmp_path / "m2.json"
    _write_artifact(a_path, "M1")
    _write_artifact(b_path, "M2")

    ds = PWMArtifactSetDataSource(
        paths=[str(a_path), str(b_path)],
        cfg_path=tmp_path / "config.yaml",
        sampling={
            "strategy": "stochastic",
            "n_sites": 4,
            "oversample_factor": 3,
            "score_threshold": -10.0,
            "score_percentile": None,
            "length_policy": "exact",
        },
    )
    entries, df = ds.load_data(rng=np.random.default_rng(0))
    assert len(entries) == 8
    assert df is not None
    assert set(df["tf"].tolist()) == {"M1", "M2"}
    assert all(len(seq) == 3 for seq in df["tfbs"].tolist())


def test_pwm_artifact_set_rejects_duplicate_ids(tmp_path: Path) -> None:
    a_path = tmp_path / "m1.json"
    b_path = tmp_path / "m2.json"
    _write_artifact(a_path, "M1")
    _write_artifact(b_path, "M1")

    ds = PWMArtifactSetDataSource(
        paths=[str(a_path), str(b_path)],
        cfg_path=tmp_path / "config.yaml",
        sampling={
            "strategy": "stochastic",
            "n_sites": 2,
            "oversample_factor": 2,
            "score_threshold": -10.0,
            "score_percentile": None,
            "length_policy": "exact",
        },
    )
    with pytest.raises(ValueError, match="Duplicate motif_id"):
        ds.load_data(rng=np.random.default_rng(1))


def test_pwm_artifact_set_overrides_by_motif_id(tmp_path: Path) -> None:
    a_path = tmp_path / "m1.json"
    b_path = tmp_path / "m2.json"
    _write_artifact(a_path, "M1")
    _write_artifact(b_path, "M2")

    ds = PWMArtifactSetDataSource(
        paths=[str(a_path), str(b_path)],
        cfg_path=tmp_path / "config.yaml",
        sampling={
            "strategy": "stochastic",
            "n_sites": 2,
            "oversample_factor": 3,
            "score_threshold": -10.0,
            "score_percentile": None,
            "length_policy": "exact",
        },
        overrides_by_motif_id={
            "M2": {
                "strategy": "stochastic",
                "n_sites": 1,
                "oversample_factor": 3,
                "score_threshold": -10.0,
                "score_percentile": None,
                "length_policy": "exact",
            }
        },
    )
    _entries, df = ds.load_data(rng=np.random.default_rng(2))
    counts = df["tf"].value_counts().to_dict()
    assert counts["M1"] == 2
    assert counts["M2"] == 1
