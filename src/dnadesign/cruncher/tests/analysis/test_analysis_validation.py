"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_analysis_validation.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dnadesign.cruncher.analysis.hits import load_baseline_hits
from dnadesign.cruncher.analysis.per_pwm import gather_per_pwm_scores
from dnadesign.cruncher.analysis.plots.summary import write_elite_topk, write_score_summary
from dnadesign.cruncher.artifacts.layout import sequences_path
from dnadesign.cruncher.core.pwm import PWM


def test_gather_per_pwm_scores_rejects_invalid_sequence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    df = pd.DataFrame(
        {
            "chain": [0],
            "draw": [0],
            "phase": ["draw"],
            "sequence": ["ACGTN"],
        }
    )
    seq_path = sequences_path(run_dir)
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(seq_path, engine="fastparquet")

    pwm_matrix = np.full((4, 4), 0.25)
    pwms = {"lexA": PWM(name="lexA", matrix=pwm_matrix)}

    with pytest.raises(ValueError, match="invalid base"):
        gather_per_pwm_scores(
            run_dir,
            change_threshold=0.1,
            pwms=pwms,
            bidirectional=False,
            scale="llr",
            out_path=run_dir / "out.csv",
        )


def test_gather_per_pwm_scores_rejects_non_positive_threshold(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    df = pd.DataFrame(
        {
            "chain": [0],
            "draw": [0],
            "phase": ["draw"],
            "sequence": ["ACGT"],
        }
    )
    seq_path = sequences_path(run_dir)
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(seq_path, engine="fastparquet")

    pwm_matrix = np.full((4, 4), 0.25)
    pwms = {"lexA": PWM(name="lexA", matrix=pwm_matrix)}

    with pytest.raises(ValueError, match="change_threshold must be > 0"):
        gather_per_pwm_scores(
            run_dir,
            change_threshold=0.0,
            pwms=pwms,
            bidirectional=False,
            scale="llr",
            out_path=run_dir / "out.csv",
        )


def test_gather_per_pwm_scores_single_draw_not_duplicated(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    df = pd.DataFrame(
        {
            "chain": [0],
            "draw": [0],
            "phase": ["draw"],
            "sequence": ["ACGTACGT"],
        }
    )
    seq_path = sequences_path(run_dir)
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(seq_path, engine="fastparquet")

    pwm_matrix = np.full((4, 4), 0.25)
    pwms = {"lexA": PWM(name="lexA", matrix=pwm_matrix)}

    out_path = run_dir / "out.csv"
    gather_per_pwm_scores(
        run_dir,
        change_threshold=0.1,
        pwms=pwms,
        bidirectional=False,
        scale="llr",
        out_path=out_path,
    )

    out_df = pd.read_csv(out_path)
    assert len(out_df) == 1
    assert out_df.loc[0, "chain"] == 0
    assert out_df.loc[0, "draw"] == 0


def test_baseline_hits_requires_columns(tmp_path: Path) -> None:
    baseline_path = tmp_path / "random_baseline_hits.parquet"
    df = pd.DataFrame({"baseline_id": [0], "tf": ["lexA"]})
    df.to_parquet(baseline_path, engine="fastparquet")
    with pytest.raises(ValueError, match="random_baseline_hits.parquet missing required columns"):
        load_baseline_hits(baseline_path)


def test_write_elite_topk_requires_sequence(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "rank": [1],
            "score_lexA": [1.0],
        }
    )
    out_path = tmp_path / "topk.csv"
    with pytest.raises(ValueError, match="sequence"):
        write_elite_topk(df, ["lexA"], out_path, top_k=1)


def test_write_score_summary_handles_pyarrow_strings(tmp_path: Path) -> None:
    score_df = pd.DataFrame({"score_lexA": [0.1, 0.2], "score_cpxR": [0.3, 0.4]})
    out_path = tmp_path / "score_summary.parquet"
    previous = pd.options.mode.string_storage
    pd.options.mode.string_storage = "pyarrow"
    try:
        write_score_summary(score_df, ["lexA", "cpxR"], out_path)
    finally:
        pd.options.mode.string_storage = previous
    assert out_path.exists()
