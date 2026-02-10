"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_trajectory_points.py

Validate trajectory point construction for plotting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dnadesign.cruncher.analysis.trajectory import (
    add_raw_llr_objective,
    build_chain_trajectory_points,
    build_trajectory_points,
    compute_best_so_far_path,
)
from dnadesign.cruncher.core.pwm import PWM


def test_build_trajectory_points_keeps_all_chains() -> None:
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
    assert "is_cold_chain" not in trajectory_df.columns


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


def test_build_trajectory_points_maps_slot_id_to_chain_when_chain_missing() -> None:
    sequences_df = pd.DataFrame(
        {
            "slot_id": [0, 0, 1, 1],
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

    assert sorted(trajectory_df["chain"].astype(int).unique()) == [0, 1]


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


def test_build_trajectory_points_carries_chain_fields() -> None:
    sequences_df = pd.DataFrame(
        {
            "chain": [0, 1, 0, 1],
            "draw": [0, 0, 1, 1],
            "sweep_idx": [0, 0, 1, 1],
            "phase": ["draw", "draw", "draw", "draw"],
            "beta": [1.0, 0.5, 1.0, 0.5],
            "score_lexA": [0.1, 0.4, 0.2, 0.5],
            "score_cpxR": [0.2, 0.3, 0.3, 0.4],
        }
    )

    trajectory_df = build_trajectory_points(
        sequences_df,
        ["lexA", "cpxR"],
        max_points=100,
        beta_ladder=[1.0, 0.5],
    )

    for required in ("chain", "beta", "sweep", "phase"):
        assert required in trajectory_df.columns
    assert sorted(trajectory_df["chain"].astype(int).unique()) == [0, 1]


def test_build_chain_trajectory_points_tracks_chain_progression() -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 1, 0, 1],
            "sweep": [0, 0, 1, 1],
            "phase": ["draw", "draw", "draw", "draw"],
            "beta": [1.0, 0.5, 1.0, 0.5],
            "x": [0.1, 0.2, 0.3, 0.4],
            "y": [0.2, 0.1, 0.4, 0.3],
            "x_metric": ["score_lexA"] * 4,
            "y_metric": ["score_cpxR"] * 4,
            "objective_scalar": [0.1, 0.2, 0.3, 0.4],
            "raw_llr_objective": [1.1, 1.2, 1.3, 1.4],
            "norm_llr_objective": [0.2, 0.3, 0.4, 0.5],
        }
    )

    chains_df = build_chain_trajectory_points(trajectory_df, max_points=100)
    chain_zero = chains_df[chains_df["chain"].astype(int) == 0].sort_values("sweep_idx")

    assert list(chain_zero["sweep_idx"].astype(int)) == [0, 1]
    assert sorted(chains_df["chain"].astype(int).unique()) == [0, 1]
    assert "x_tf" in chains_df.columns
    assert "y_tf" in chains_df.columns


def test_add_raw_llr_objective_adds_combined_raw_llr_column() -> None:
    trajectory_df = pd.DataFrame(
        {
            "sequence": ["AACCAA", "TTGGTT"],
            "chain": [0, 1],
            "sweep_idx": [0, 0],
        }
    )
    pwms = {
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

    enriched = add_raw_llr_objective(
        trajectory_df,
        ["lexA", "cpxR"],
        pwms=pwms,
        objective_config={"combine": "min", "softmin": {"enabled": False}},
        bidirectional=True,
        pwm_pseudocounts=0.10,
        log_odds_clip=None,
    )

    assert "raw_llr_lexA" in enriched.columns
    assert "raw_llr_cpxR" in enriched.columns
    assert "norm_llr_lexA" in enriched.columns
    assert "norm_llr_cpxR" in enriched.columns
    assert "raw_llr_objective" in enriched.columns
    assert "norm_llr_objective" in enriched.columns
    assert enriched["raw_llr_objective"].notna().all()
    assert enriched["norm_llr_objective"].notna().all()
