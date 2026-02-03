"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_pwm_artifact_set_source.py

PWM artifact-set data source sampling tests.

Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dnadesign.densegen.src.adapters.sources import PWMArtifactSetDataSource
from dnadesign.densegen.src.adapters.sources.pwm_sampling import build_log_odds
from dnadesign.densegen.src.integrations.meme_suite import resolve_executable
from dnadesign.densegen.tests.pwm_sampling_fixtures import (
    fixed_candidates_mining,
    sampling_config,
    selection_mmr,
)

_FIMO_MISSING = resolve_executable("fimo", tool_path=None) is None


def _write_artifact(path: Path, motif_id: str, *, length: int = 3) -> None:
    rows = [
        {"A": 0.7, "C": 0.1, "G": 0.1, "T": 0.1},
        {"A": 0.1, "C": 0.7, "G": 0.1, "T": 0.1},
        {"A": 0.1, "C": 0.1, "G": 0.7, "T": 0.1},
        {"A": 0.1, "C": 0.1, "G": 0.1, "T": 0.7},
    ]
    matrix = [rows[idx % len(rows)] for idx in range(int(length))]
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


@pytest.mark.skipif(
    _FIMO_MISSING,
    reason="fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).",
)
def test_pwm_artifact_set_sampling(tmp_path: Path) -> None:
    a_path = tmp_path / "m1.json"
    b_path = tmp_path / "m2.json"
    _write_artifact(a_path, "M1")
    _write_artifact(b_path, "M2")

    ds = PWMArtifactSetDataSource(
        paths=[str(a_path), str(b_path)],
        cfg_path=tmp_path / "config.yaml",
        sampling=sampling_config(
            n_sites=4,
            strategy="stochastic",
            mining=fixed_candidates_mining(batch_size=10, candidates=80),
            length_policy="exact",
        ),
    )
    entries, df, _summaries = ds.load_data(rng=np.random.default_rng(0))
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
        sampling=sampling_config(
            n_sites=2,
            strategy="stochastic",
            mining=fixed_candidates_mining(batch_size=10, candidates=40),
            length_policy="exact",
        ),
    )
    with pytest.raises(ValueError, match="Duplicate motif_id"):
        ds.load_data(rng=np.random.default_rng(1))


@pytest.mark.skipif(
    _FIMO_MISSING,
    reason="fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).",
)
def test_pwm_artifact_set_overrides_by_motif_id(tmp_path: Path) -> None:
    a_path = tmp_path / "m1.json"
    b_path = tmp_path / "m2.json"
    _write_artifact(a_path, "M1")
    _write_artifact(b_path, "M2")

    ds = PWMArtifactSetDataSource(
        paths=[str(a_path), str(b_path)],
        cfg_path=tmp_path / "config.yaml",
        sampling=sampling_config(
            n_sites=2,
            strategy="stochastic",
            mining=fixed_candidates_mining(batch_size=10, candidates=60),
            length_policy="exact",
        ),
        overrides_by_motif_id={
            "M2": sampling_config(
                n_sites=1,
                strategy="stochastic",
                mining=fixed_candidates_mining(batch_size=10, candidates=30),
                length_policy="exact",
            )
        },
    )
    _entries, df, _summaries = ds.load_data(rng=np.random.default_rng(2))
    counts = df["tf"].value_counts().to_dict()
    assert counts["M1"] == 2
    assert counts["M2"] == 1


def test_pwm_artifact_set_prevalidates_mmr_range(tmp_path: Path, monkeypatch) -> None:
    a_path = tmp_path / "m1.json"
    b_path = tmp_path / "m2.json"
    _write_artifact(a_path, "M1", length=2)
    _write_artifact(b_path, "M2", length=3)

    def _should_not_run(*_args, **_kwargs):
        raise RuntimeError("sample_pwm_sites should not run before prevalidation")

    monkeypatch.setattr(
        "dnadesign.densegen.src.adapters.sources.pwm_artifact_set.sample_pwm_sites",
        _should_not_run,
    )

    ds = PWMArtifactSetDataSource(
        paths=[str(a_path), str(b_path)],
        cfg_path=tmp_path / "config.yaml",
        sampling=sampling_config(
            n_sites=2,
            strategy="stochastic",
            mining=fixed_candidates_mining(batch_size=10, candidates=40),
            selection=selection_mmr(alpha=0.5),
            length_policy="range",
            length_range=(2, 3),
        ),
    )
    with pytest.raises(ValueError, match="MMR requires a fixed trim window"):
        ds.load_data(rng=np.random.default_rng(3))
