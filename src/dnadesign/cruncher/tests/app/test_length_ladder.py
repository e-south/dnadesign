"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_length_ladder.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dnadesign.cruncher.app.sample.auto_opt import _warm_start_seeds_from_elites
from dnadesign.cruncher.app.sample_workflow import _candidate_lengths, _warm_start_seeds_from_sequences
from dnadesign.cruncher.artifacts.layout import elites_path, sequences_path
from dnadesign.cruncher.config.schema_v2 import (
    AutoOptConfig,
    AutoOptLengthConfig,
    InitConfig,
    SampleBudgetConfig,
    SampleConfig,
)
from dnadesign.cruncher.core.pwm import PWM


def _pwm(name: str, length: int) -> PWM:
    matrix = np.full((length, 4), 0.25)
    return PWM(name=name, matrix=matrix)


def test_candidate_lengths_ladder_defaults() -> None:
    sample_cfg = SampleConfig(
        budget=SampleBudgetConfig(tune=1, draws=1, restarts=1),
        init=InitConfig(kind="random", length=12),
    )
    auto_cfg = AutoOptConfig(length=AutoOptLengthConfig(enabled=True, mode="ladder", step=1))
    pwms = {"tfA": _pwm("tfA", 5), "tfB": _pwm("tfB", 3)}
    lengths = _candidate_lengths(sample_cfg, auto_cfg, pwms)
    assert lengths == [5, 6, 7, 8]


def test_warm_start_seeds_extend_sequences(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    seq_file = sequences_path(run_dir)
    seq_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"sequence": ["ACG", "TGCA"], "score_tfA": [1.0, 0.5]})
    df.to_parquet(seq_file, engine="fastparquet")

    rng = np.random.default_rng(0)
    seeds = _warm_start_seeds_from_sequences(
        run_dir,
        target_length=4,
        rng=rng,
        pad_with="A",
        max_seeds=None,
    )
    assert seeds
    assert all(seed.size == 4 for seed in seeds)
    for seed in seeds:
        assert np.all((seed >= 0) & (seed <= 3))


def test_warm_start_requires_sequences_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing_sequences"
    rng = np.random.default_rng(0)
    with pytest.raises(FileNotFoundError, match="sequences.parquet"):
        _warm_start_seeds_from_sequences(
            run_dir,
            target_length=4,
            rng=rng,
            pad_with="A",
            max_seeds=None,
        )


def test_warm_start_requires_matching_sequences(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    seq_file = sequences_path(run_dir)
    seq_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"sequence": ["AC", "TGCAAA"], "score_tfA": [1.0, 0.5]})
    df.to_parquet(seq_file, engine="fastparquet")

    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Warm-start found no usable sequences"):
        _warm_start_seeds_from_sequences(
            run_dir,
            target_length=4,
            rng=rng,
            pad_with="A",
            max_seeds=None,
        )


def test_warm_start_elites_require_sequence_column(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    elite_file = elites_path(run_dir)
    elite_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"rank": [1, 2], "score_tfA": [1.0, 0.5]})
    df.to_parquet(elite_file, engine="fastparquet")

    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="sequence"):
        _warm_start_seeds_from_elites(
            run_dir,
            target_length=4,
            rng=rng,
            pad_with="A",
            max_seeds=None,
        )


def test_warm_start_elites_require_matching_sequences(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    elite_file = elites_path(run_dir)
    elite_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"sequence": ["AC", "TGCAAA"], "rank": [1, 2]})
    df.to_parquet(elite_file, engine="fastparquet")

    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Warm-start found no usable elite sequences"):
        _warm_start_seeds_from_elites(
            run_dir,
            target_length=4,
            rng=rng,
            pad_with="A",
            max_seeds=None,
        )
