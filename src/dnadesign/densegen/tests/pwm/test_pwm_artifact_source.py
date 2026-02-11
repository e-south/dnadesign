"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/pwm/test_pwm_artifact_source.py

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
from dnadesign.densegen.src.adapters.sources.stage_a.stage_a_metadata import TFBSMeta
from dnadesign.densegen.src.adapters.sources.stage_a.stage_a_types import SelectionMeta
from dnadesign.densegen.src.integrations.meme_suite import resolve_executable
from dnadesign.densegen.tests.pwm_sampling_fixtures import fixed_candidates_mining, sampling_config

_FIMO_MISSING = resolve_executable("fimo", tool_path=None) is None


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


@pytest.mark.skipif(
    _FIMO_MISSING,
    reason="fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).",
)
def test_pwm_artifact_sampling_exact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "motif.json"
    _write_artifact(artifact_path)
    ds = PWMArtifactDataSource(
        path=str(artifact_path),
        cfg_path=tmp_path / "config.yaml",
        input_name="demo_input",
        sampling=sampling_config(
            n_sites=5,
            strategy="stochastic",
            mining=fixed_candidates_mining(batch_size=10, candidates=60),
            length_policy="exact",
        ),
    )
    entries, df, _summaries = ds.load_data(rng=np.random.default_rng(0))
    assert len(entries) == 5
    assert df is not None
    assert set(df["tf"].tolist()) == {"M1"}
    assert all(len(seq) == 3 for seq in df["tfbs"].tolist())


@pytest.mark.skipif(
    _FIMO_MISSING,
    reason="fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).",
)
def test_pwm_artifact_sampling_range(tmp_path: Path) -> None:
    artifact_path = tmp_path / "motif.json"
    _write_artifact(artifact_path)
    ds = PWMArtifactDataSource(
        path=str(artifact_path),
        cfg_path=tmp_path / "config.yaml",
        input_name="demo_input",
        sampling=sampling_config(
            n_sites=6,
            strategy="stochastic",
            mining=fixed_candidates_mining(batch_size=10, candidates=80),
            length_policy="range",
            length_range=(3, 5),
        ),
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
        sampling=sampling_config(
            n_sites=2,
            strategy="stochastic",
            mining=fixed_candidates_mining(batch_size=10, candidates=20),
            length_policy="exact",
        ),
    )
    with pytest.raises(ValueError, match="log_odds\\[0\\]\\[A\\].*finite"):
        ds.load_data(rng=np.random.default_rng(0))


@pytest.mark.skipif(
    _FIMO_MISSING,
    reason="fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).",
)
def test_pwm_sampling_shortfall_includes_motif_id(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    artifact_path = tmp_path / "motif.json"
    _write_artifact(artifact_path)
    ds = PWMArtifactDataSource(
        path=str(artifact_path),
        cfg_path=tmp_path / "config.yaml",
        input_name="demo_input",
        sampling=sampling_config(
            n_sites=2,
            strategy="stochastic",
            mining=fixed_candidates_mining(batch_size=2, candidates=1, log_every_batches=1),
            length_policy="exact",
        ),
    )
    with caplog.at_level("WARNING"):
        ds.load_data(rng=np.random.default_rng(1))
    assert "motif 'M1'" in caplog.text
    assert "shortfall" in caplog.text.lower()


def test_pwm_artifact_sampling_generates_tfbs_rows_without_fimo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "motif.json"
    _write_artifact(artifact_path)

    def _fake_sample_pwm_sites(_rng, _motif, **_kwargs):
        selected = ["ACG"]
        meta = TFBSMeta(
            best_hit_score=6.5,
            rank_within_regulator=1,
            tier=0,
            fimo_start=5,
            fimo_stop=8,
            fimo_strand="+",
            tfbs_core="ACG",
            fimo_matched_sequence="ACG",
            selection_meta=SelectionMeta(selection_rank=1, selection_utility=1.0, selection_score_norm=0.95),
            selection_policy="top_score",
            selection_alpha=None,
            selection_similarity=None,
            selection_relevance_norm=None,
            selection_pool_size_final=1,
            selection_pool_rung_fraction_used=1.0,
            selection_pool_min_score_norm_used=None,
            selection_pool_capped=False,
            selection_pool_cap_value=None,
            selection_pool_target_size=1,
            selection_pool_degenerate=True,
            tier_target_fraction=0.01,
            tier_target_required_unique=100,
            tier_target_met=True,
            tier_target_eligible_unique=1,
        )
        return selected, {"ACG": meta}, None

    monkeypatch.setattr(
        "dnadesign.densegen.src.adapters.sources.pwm_artifact.sample_pwm_sites",
        _fake_sample_pwm_sites,
    )

    ds = PWMArtifactDataSource(
        path=str(artifact_path),
        cfg_path=tmp_path / "config.yaml",
        input_name="demo_input",
        sampling=sampling_config(
            n_sites=1,
            strategy="stochastic",
            mining=fixed_candidates_mining(batch_size=10, candidates=20),
            length_policy="exact",
        ),
    )
    entries, df, _summaries = ds.load_data(rng=np.random.default_rng(0))

    assert entries == [("M1", "ACG", str(artifact_path))]
    assert df is not None
    assert len(df) == 1
    assert df.loc[0, "tf"] == "M1"
    assert df.loc[0, "tfbs"] == "ACG"
    assert df.loc[0, "best_hit_score"] == 6.5
    assert df.loc[0, "selection_rank"] == 1
    assert df.loc[0, "tier"] == 0
    assert isinstance(df.loc[0, "tfbs_id"], str)
    assert df.loc[0, "tfbs_id"]
