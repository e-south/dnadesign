"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_pwm_artifact_source.py

PWM artifact data source sampling tests.

Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dnadesign.densegen.src.adapters.sources import PWMArtifactDataSource
from dnadesign.densegen.src.adapters.sources.pwm_sampling import build_log_odds


def _write_artifact(path: Path) -> None:
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
        "motif_id": "M1",
        "alphabet": "ACGT",
        "matrix_semantics": "probabilities",
        "background": background,
        "probabilities": matrix,
        "log_odds": log_odds,
    }
    path.write_text(json.dumps(artifact))


def test_pwm_artifact_sampling_exact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "motif.json"
    _write_artifact(artifact_path)
    ds = PWMArtifactDataSource(
        path=str(artifact_path),
        cfg_path=tmp_path / "config.yaml",
        input_name="demo_input",
        sampling={
            "strategy": "stochastic",
            "n_sites": 5,
            "oversample_factor": 3,
            "score_threshold": -10.0,
            "score_percentile": None,
            "length_policy": "exact",
        },
    )
    entries, df, _summaries = ds.load_data(rng=np.random.default_rng(0))
    assert len(entries) == 5
    assert df is not None
    assert set(df["tf"].tolist()) == {"M1"}
    assert all(len(seq) == 3 for seq in df["tfbs"].tolist())


def test_pwm_artifact_sampling_range(tmp_path: Path) -> None:
    artifact_path = tmp_path / "motif.json"
    _write_artifact(artifact_path)
    ds = PWMArtifactDataSource(
        path=str(artifact_path),
        cfg_path=tmp_path / "config.yaml",
        input_name="demo_input",
        sampling={
            "strategy": "stochastic",
            "n_sites": 6,
            "oversample_factor": 3,
            "score_threshold": -10.0,
            "score_percentile": None,
            "length_policy": "range",
            "length_range": (3, 5),
        },
    )
    entries, df, _summaries = ds.load_data(rng=np.random.default_rng(1))
    assert len(entries) == 6
    assert df is not None
    lengths = [len(seq) for seq in df["tfbs"].tolist()]
    assert min(lengths) >= 3
    assert max(lengths) <= 5


def test_pwm_artifact_rejects_nonfinite_log_odds(tmp_path: Path) -> None:
    artifact_path = tmp_path / "motif.json"
    matrix = [
        {"A": 0.7, "C": 0.1, "G": 0.1, "T": 0.1},
        {"A": 0.1, "C": 0.7, "G": 0.1, "T": 0.1},
    ]
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    log_odds = build_log_odds(matrix, background)
    log_odds[0]["A"] = float("inf")
    artifact = {
        "schema_version": "1.0",
        "producer": "test",
        "motif_id": "M1",
        "alphabet": "ACGT",
        "matrix_semantics": "probabilities",
        "background": background,
        "probabilities": matrix,
        "log_odds": log_odds,
    }
    artifact_path.write_text(json.dumps(artifact))
    ds = PWMArtifactDataSource(
        path=str(artifact_path),
        cfg_path=tmp_path / "config.yaml",
        input_name="demo_input",
        sampling={
            "strategy": "stochastic",
            "n_sites": 2,
            "oversample_factor": 2,
            "score_threshold": -10.0,
            "score_percentile": None,
            "length_policy": "exact",
        },
    )
    with pytest.raises(ValueError, match="log_odds\\[0\\]\\[A\\].*finite"):
        ds.load_data(rng=np.random.default_rng(0))


def test_pwm_sampling_shortfall_includes_motif_id(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    artifact_path = tmp_path / "motif.json"
    _write_artifact(artifact_path)
    ds = PWMArtifactDataSource(
        path=str(artifact_path),
        cfg_path=tmp_path / "config.yaml",
        input_name="demo_input",
        sampling={
            "strategy": "stochastic",
            "n_sites": 2,
            "oversample_factor": 1,
            "score_threshold": 100.0,
            "score_percentile": None,
            "length_policy": "exact",
        },
    )
    with caplog.at_level("WARNING"):
        ds.load_data(rng=np.random.default_rng(1))
    assert "motif 'M1'" in caplog.text
    assert "shortfall" in caplog.text.lower()
