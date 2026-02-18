"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/pwm/test_pwm_meme_set_source.py

PWM MEME set data source sampling tests.

Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dnadesign.densegen.src.adapters.sources import PWMMemeSetDataSource
from dnadesign.densegen.src.adapters.sources import pwm_meme_set as pwm_meme_set_module
from dnadesign.densegen.src.integrations.meme_suite import resolve_executable
from dnadesign.densegen.tests.pwm_sampling_fixtures import fixed_candidates_mining, sampling_config

pytestmark = pytest.mark.fimo

_FIMO_MISSING = resolve_executable("fimo", tool_path=None) is None


def _meme_text(motif_id: str) -> str:
    return f"""MEME version 4

ALPHABET= ACGT

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

MOTIF {motif_id}
letter-probability matrix: alength= 4 w= 3 nsites= 20 E= 0
0.8 0.1 0.05 0.05
0.1 0.7 0.1 0.1
0.1 0.1 0.7 0.1
"""


@pytest.mark.skipif(
    _FIMO_MISSING,
    reason="fimo executable not available (run tests via `pixi run pytest` or set MEME_BIN).",
)
def test_pwm_meme_set_sampling(tmp_path: Path) -> None:
    meme_a = tmp_path / "lexA.meme"
    meme_b = tmp_path / "cpxR.meme"
    meme_a.write_text(_meme_text("lexA"))
    meme_b.write_text(_meme_text("cpxR"))

    ds = PWMMemeSetDataSource(
        paths=[str(meme_a), str(meme_b)],
        cfg_path=tmp_path / "config.yaml",
        input_name="demo_input",
        motif_ids=["lexA", "cpxR"],
        sampling=sampling_config(
            n_sites=3,
            strategy="stochastic",
            mining=fixed_candidates_mining(batch_size=10, candidates=60),
        ),
    )
    entries, df, _summaries = ds.load_data(rng=np.random.default_rng(0))
    assert len(entries) == 6
    assert set(df["tf"].tolist()) == {"lexA", "cpxR"}


def test_pwm_meme_set_duplicate_motif_ids(tmp_path: Path) -> None:
    meme_a = tmp_path / "motif_a.meme"
    meme_b = tmp_path / "motif_b.meme"
    meme_a.write_text(_meme_text("dup"))
    meme_b.write_text(_meme_text("dup"))

    ds = PWMMemeSetDataSource(
        paths=[str(meme_a), str(meme_b)],
        cfg_path=tmp_path / "config.yaml",
        input_name="demo_input",
        motif_ids=None,
        sampling=sampling_config(
            n_sites=1,
            strategy="stochastic",
            mining=fixed_candidates_mining(batch_size=10, candidates=10),
        ),
    )
    with pytest.raises(ValueError, match="Duplicate motif_id"):
        ds.load_data(rng=np.random.default_rng(1))


def test_pwm_meme_set_requires_explicit_collision_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    meme_a = tmp_path / "lexA.meme"
    meme_a.write_text(_meme_text("lexA"))

    original_sampling_kwargs = pwm_meme_set_module.sampling_kwargs_from_config

    def _sampling_kwargs_missing_collision_mode(sampling):
        kwargs = original_sampling_kwargs(sampling)
        kwargs.pop("cross_regulator_core_collisions", None)
        return kwargs

    def _should_not_run(*_args, **_kwargs):
        raise RuntimeError("sample_pwm_sites should not run when collision mode is missing")

    monkeypatch.setattr(pwm_meme_set_module, "sampling_kwargs_from_config", _sampling_kwargs_missing_collision_mode)
    monkeypatch.setattr(pwm_meme_set_module, "sample_pwm_sites", _should_not_run)

    ds = PWMMemeSetDataSource(
        paths=[str(meme_a)],
        cfg_path=tmp_path / "config.yaml",
        input_name="demo_input",
        motif_ids=["lexA"],
        sampling=sampling_config(
            n_sites=1,
            strategy="stochastic",
            mining=fixed_candidates_mining(batch_size=10, candidates=20),
        ),
    )
    with pytest.raises(ValueError, match="cross_regulator_core_collisions"):
        ds.load_data(rng=np.random.default_rng(2))
