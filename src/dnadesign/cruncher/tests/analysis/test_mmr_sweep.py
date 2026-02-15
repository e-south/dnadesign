"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_mmr_sweep.py

Tests for artifact-driven MMR sweep replay diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dnadesign.cruncher.analysis.mmr_sweep import run_mmr_sweep
from dnadesign.cruncher.analysis.parquet import write_parquet
from dnadesign.cruncher.core.pwm import PWM


def _pwms() -> dict[str, PWM]:
    return {
        "lexA": PWM(
            name="lexA",
            matrix=np.asarray(
                [
                    [0.90, 0.03, 0.03, 0.04],
                    [0.04, 0.90, 0.03, 0.03],
                ],
                dtype=float,
            ),
        ),
        "cpxR": PWM(
            name="cpxR",
            matrix=np.asarray(
                [
                    [0.03, 0.03, 0.90, 0.04],
                    [0.03, 0.04, 0.03, 0.90],
                ],
                dtype=float,
            ),
        ),
    }


def test_run_mmr_sweep_emits_grid_metrics() -> None:
    sequences_df = pd.DataFrame(
        {
            "chain": [0, 0, 1, 1],
            "sweep_idx": [0, 1, 0, 1],
            "phase": ["draw", "draw", "draw", "draw"],
            "sequence": ["AACCAA", "AATCAA", "TTGGTT", "TTGGAT"],
            "score_lexA": [0.45, 0.46, 0.42, 0.41],
            "score_cpxR": [0.43, 0.44, 0.40, 0.39],
        }
    )
    elites_df = pd.DataFrame({"chain": [0, 1], "draw_idx": [1, 1]})

    sweep_df = run_mmr_sweep(
        sequences_df=sequences_df,
        elites_df=elites_df,
        tf_names=["lexA", "cpxR"],
        pwms=_pwms(),
        objective_config={"combine": "min", "total_sweeps": 2, "softmin": {"enabled": False}},
        bidirectional=True,
        elite_k=2,
        pwm_pseudocounts=0.10,
        log_odds_clip=None,
        pool_size_values=["auto", 2],
        diversity_values=[0.0],
        baseline_pool_size="auto",
        baseline_diversity=None,
    )

    assert not sweep_df.empty
    assert len(sweep_df) == 2
    assert "mean_pairwise_core_distance" in sweep_df.columns
    assert "mean_pairwise_full_distance" in sweep_df.columns
    assert "score_weight" in sweep_df.columns
    assert "diversity_weight" in sweep_df.columns
    assert "diversity" in sweep_df.columns
    assert set(sweep_df["distance_metric"].unique()) == {"none"}
    assert float(sweep_df["score_weight"].iloc[0]) == 1.0
    assert float(sweep_df["diversity_weight"].iloc[0]) == 0.0
    assert set(sweep_df["constraint_policy"].unique()) == {"disabled"}
    assert "jaccard_vs_current_elites" in sweep_df.columns
    assert sweep_df["candidate_count"].min() >= 0
    assert bool(sweep_df["is_current_config"].any())


def test_run_mmr_sweep_table_is_parquet_writable_with_mixed_pool_size_inputs(tmp_path) -> None:
    sequences_df = pd.DataFrame(
        {
            "chain": [0, 0, 1, 1],
            "sweep_idx": [0, 1, 0, 1],
            "phase": ["draw", "draw", "draw", "draw"],
            "sequence": ["AACCAA", "AATCAA", "TTGGTT", "TTGGAT"],
            "score_lexA": [0.45, 0.46, 0.42, 0.41],
            "score_cpxR": [0.43, 0.44, 0.40, 0.39],
        }
    )
    elites_df = pd.DataFrame({"chain": [0, 1], "draw_idx": [1, 1]})

    sweep_df = run_mmr_sweep(
        sequences_df=sequences_df,
        elites_df=elites_df,
        tf_names=["lexA", "cpxR"],
        pwms=_pwms(),
        objective_config={"combine": "min", "total_sweeps": 2, "softmin": {"enabled": False}},
        bidirectional=True,
        elite_k=2,
        pwm_pseudocounts=0.10,
        log_odds_clip=None,
        pool_size_values=["auto", 2],
        diversity_values=[0.0],
        baseline_pool_size="auto",
        baseline_diversity=None,
    )

    out_path = tmp_path / "mmr_sweep.parquet"
    write_parquet(sweep_df, out_path)
    assert out_path.exists()


def test_run_mmr_sweep_supports_all_pool_baseline_marker() -> None:
    sequences_df = pd.DataFrame(
        {
            "chain": [0, 0, 1, 1],
            "sweep_idx": [0, 1, 0, 1],
            "phase": ["draw", "draw", "draw", "draw"],
            "sequence": ["AACCAA", "AATCAA", "TTGGTT", "TTGGAT"],
            "score_lexA": [0.45, 0.46, 0.42, 0.41],
            "score_cpxR": [0.43, 0.44, 0.40, 0.39],
        }
    )
    elites_df = pd.DataFrame({"chain": [0, 1], "draw_idx": [1, 1]})

    sweep_df = run_mmr_sweep(
        sequences_df=sequences_df,
        elites_df=elites_df,
        tf_names=["lexA", "cpxR"],
        pwms=_pwms(),
        objective_config={"combine": "min", "total_sweeps": 2, "softmin": {"enabled": False}},
        bidirectional=True,
        elite_k=2,
        pwm_pseudocounts=0.10,
        log_odds_clip=None,
        pool_size_values=["all"],
        diversity_values=[0.0],
        baseline_pool_size="all",
        baseline_diversity=0.0,
    )

    assert len(sweep_df) == 1
    assert bool(sweep_df["is_current_config"].iloc[0])
