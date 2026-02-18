"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/pwm/test_pwm_other_sources.py

PWM sampling tests for JASPAR and matrix CSV sources.

Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dnadesign.densegen.src.adapters.sources import PWMJasparDataSource, PWMMatrixCSVDataSource
from dnadesign.densegen.src.adapters.sources import pwm_jaspar as pwm_jaspar_module
from dnadesign.densegen.src.adapters.sources import pwm_matrix_csv as pwm_matrix_csv_module
from dnadesign.densegen.src.integrations.meme_suite import resolve_executable
from dnadesign.densegen.tests.pwm_sampling_fixtures import fixed_candidates_mining, sampling_config

pytestmark = pytest.mark.fimo

_FIMO_MISSING = resolve_executable("fimo", tool_path=None) is None

JASPAR_TEXT = """>M1
A [ 3 1 0 ]
C [ 0 3 1 ]
G [ 1 0 3 ]
T [ 0 1 0 ]
"""


@pytest.mark.skipif(
    _FIMO_MISSING,
    reason="fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).",
)
def test_pwm_jaspar_sampling(tmp_path: Path) -> None:
    jaspar_path = tmp_path / "motifs.pfm"
    jaspar_path.write_text(JASPAR_TEXT)
    ds = PWMJasparDataSource(
        path=str(jaspar_path),
        cfg_path=tmp_path / "config.yaml",
        input_name="demo_input",
        motif_ids=["M1"],
        sampling=sampling_config(
            n_sites=4,
            strategy="stochastic",
            mining=fixed_candidates_mining(batch_size=10, candidates=80),
        ),
    )
    entries, df, _summaries = ds.load_data(rng=np.random.default_rng(0))
    assert len(entries) == 4
    assert set(df["tf"].tolist()) == {"M1"}


@pytest.mark.skipif(
    _FIMO_MISSING,
    reason="fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).",
)
def test_pwm_matrix_csv_sampling(tmp_path: Path) -> None:
    csv_path = tmp_path / "matrix.csv"
    csv_path.write_text("A,C,G,T\n3,0,1,0\n1,3,0,1\n0,1,3,0\n")
    ds = PWMMatrixCSVDataSource(
        path=str(csv_path),
        cfg_path=tmp_path / "config.yaml",
        input_name="demo_input",
        motif_id="M1",
        columns={"A": "A", "C": "C", "G": "G", "T": "T"},
        sampling=sampling_config(
            n_sites=1,
            strategy="consensus",
            mining=fixed_candidates_mining(batch_size=10, candidates=10),
        ),
    )
    entries, df, _summaries = ds.load_data(rng=np.random.default_rng(1))
    assert len(entries) == 1
    assert df["tf"].unique().tolist() == ["M1"]


def test_pwm_jaspar_requires_explicit_collision_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jaspar_path = tmp_path / "motifs.pfm"
    jaspar_path.write_text(JASPAR_TEXT)

    original_sampling_kwargs = pwm_jaspar_module.sampling_kwargs_from_config

    def _sampling_kwargs_missing_collision_mode(sampling):
        kwargs = original_sampling_kwargs(sampling)
        kwargs.pop("cross_regulator_core_collisions", None)
        return kwargs

    def _should_not_run(*_args, **_kwargs):
        raise RuntimeError("sample_pwm_sites should not run when collision mode is missing")

    monkeypatch.setattr(pwm_jaspar_module, "sampling_kwargs_from_config", _sampling_kwargs_missing_collision_mode)
    monkeypatch.setattr(pwm_jaspar_module, "sample_pwm_sites", _should_not_run)

    ds = PWMJasparDataSource(
        path=str(jaspar_path),
        cfg_path=tmp_path / "config.yaml",
        input_name="demo_input",
        motif_ids=["M1"],
        sampling=sampling_config(
            n_sites=1,
            strategy="stochastic",
            mining=fixed_candidates_mining(batch_size=10, candidates=20),
        ),
    )
    with pytest.raises(ValueError, match="cross_regulator_core_collisions"):
        ds.load_data(rng=np.random.default_rng(3))


def test_pwm_matrix_csv_requires_explicit_length_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "matrix.csv"
    csv_path.write_text("A,C,G,T\n3,0,1,0\n1,3,0,1\n0,1,3,0\n")

    original_sampling_kwargs = pwm_matrix_csv_module.sampling_kwargs_from_config

    def _sampling_kwargs_missing_length_policy(sampling):
        kwargs = original_sampling_kwargs(sampling)
        kwargs.pop("length_policy", None)
        return kwargs

    def _should_not_run(*_args, **_kwargs):
        raise RuntimeError("sample_pwm_sites should not run when length policy is missing")

    monkeypatch.setattr(pwm_matrix_csv_module, "sampling_kwargs_from_config", _sampling_kwargs_missing_length_policy)
    monkeypatch.setattr(pwm_matrix_csv_module, "sample_pwm_sites", _should_not_run)

    ds = PWMMatrixCSVDataSource(
        path=str(csv_path),
        cfg_path=tmp_path / "config.yaml",
        input_name="demo_input",
        motif_id="M1",
        columns={"A": "A", "C": "C", "G": "G", "T": "T"},
        sampling=sampling_config(
            n_sites=1,
            strategy="consensus",
            mining=fixed_candidates_mining(batch_size=10, candidates=10),
        ),
    )
    with pytest.raises(ValueError, match="length.policy"):
        ds.load_data(rng=np.random.default_rng(4))
