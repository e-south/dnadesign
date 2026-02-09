"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_trajectory_points.py

Validate trajectory point construction for plotting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd
import pytest

from dnadesign.cruncher.analysis.trajectory import build_trajectory_points, compute_best_so_far_path


def test_build_trajectory_points_keeps_all_chains_and_marks_cold() -> None:
    sequences_df = pd.DataFrame(
        {
            "chain": [0, 0, 1, 1],
            "draw": [0, 1, 0, 1],
            "phase": ["tune", "draw", "tune", "draw"],
            "score_lexA": [0.1, 0.2, 0.3, 0.4],
            "score_cpxR": [0.2, 0.3, 0.1, 0.2],
        }
    )

    trajectory_df = build_trajectory_points(
        sequences_df,
        ["lexA", "cpxR"],
        max_points=100,
        beta_ladder=[1.0, 0.5],
    )

    assert set(trajectory_df["chain"].astype(int).unique()) == {0, 1}
    cold_rows = trajectory_df[trajectory_df["chain"] == 0]
    hot_rows = trajectory_df[trajectory_df["chain"] == 1]
    assert set(cold_rows["is_cold_chain"].astype(int).unique()) == {1}
    assert set(hot_rows["is_cold_chain"].astype(int).unique()) == {0}


def test_build_trajectory_points_subsample_keeps_dense_early_sweeps() -> None:
    draws = list(range(100))
    sequences_df = pd.DataFrame(
        {
            "chain": [0] * 100,
            "draw": draws,
            "phase": ["draw"] * 100,
            "score_lexA": [0.5 + (i * 0.001) for i in draws],
            "score_cpxR": [0.3 + (i * 0.001) for i in draws],
        }
    )

    trajectory_df = build_trajectory_points(
        sequences_df,
        ["lexA", "cpxR"],
        max_points=10,
        beta_ladder=[1.0],
    )

    assert len(trajectory_df) == 10
    assert list(trajectory_df["sweep"].astype(int).head(4)) == [0, 1, 2, 3]


def test_build_trajectory_points_uses_max_beta_as_cold_chain() -> None:
    sequences_df = pd.DataFrame(
        {
            "chain": [0, 0, 1, 1],
            "draw": [0, 1, 0, 1],
            "phase": ["draw", "draw", "draw", "draw"],
            "score_lexA": [0.1, 0.2, 0.8, 0.9],
            "score_cpxR": [0.2, 0.3, 0.7, 0.8],
        }
    )

    trajectory_df = build_trajectory_points(
        sequences_df,
        ["lexA", "cpxR"],
        max_points=100,
        beta_ladder=[0.2, 1.0],
    )

    cold_rows = trajectory_df[trajectory_df["chain"] == 1]
    hot_rows = trajectory_df[trajectory_df["chain"] == 0]
    assert set(cold_rows["is_cold_chain"].astype(int).unique()) == {1}
    assert set(hot_rows["is_cold_chain"].astype(int).unique()) == {0}


def test_build_trajectory_points_rejects_ambiguous_cold_chain_beta_ties() -> None:
    sequences_df = pd.DataFrame(
        {
            "chain": [0, 0, 1, 1],
            "draw": [0, 1, 0, 1],
            "phase": ["draw", "draw", "draw", "draw"],
            "score_lexA": [0.2, 0.3, 0.5, 0.6],
            "score_cpxR": [0.1, 0.2, 0.4, 0.5],
        }
    )

    with pytest.raises(ValueError, match="multiple chains share the maximum beta"):
        build_trajectory_points(
            sequences_df,
            ["lexA", "cpxR"],
            max_points=100,
            beta_ladder=[1.0, 1.0],
        )


def test_compute_best_so_far_path_is_monotonic() -> None:
    trajectory_df = pd.DataFrame(
        {
            "sweep": [0, 0, 1, 1, 2, 2],
            "x": [0.1, 0.2, 0.3, 0.25, 0.4, 0.35],
            "y": [0.2, 0.1, 0.4, 0.2, 0.45, 0.3],
            "objective_scalar": [0.1, 0.2, 0.15, 0.25, 0.22, 0.3],
        }
    )

    best_path = compute_best_so_far_path(trajectory_df, objective_col="objective_scalar")

    assert list(best_path["sweep"].astype(int)) == [0, 1, 2]
    assert best_path["objective_scalar"].is_monotonic_increasing
