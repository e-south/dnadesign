"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_trajectory_score_space_boundaries.py

Characterization tests for trajectory score-space helper boundaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from dnadesign.cruncher.analysis.plots.trajectory_score_space import (
    _prepare_elite_points,
    _resolve_elite_match_column,
    _sample_scatter_backbone,
    _sample_scatter_best_progression,
)


def test_resolve_elite_match_column_prefers_higher_overlap() -> None:
    trajectory_df = pd.DataFrame(
        {
            "sequence_hash": ["hash-1", "hash-2", "hash-3"],
            "sequence": ["seq-1", "seq-2", "seq-3"],
        }
    )
    elites_df = pd.DataFrame(
        {
            "sequence_hash": ["hash-2", "hash-3", "hash-9"],
            "sequence": ["seq-1", "seq-8", "seq-9"],
        }
    )

    selected = _resolve_elite_match_column(trajectory_df, elites_df)

    assert selected == "sequence_hash"


def test_prepare_elite_points_exact_maps_and_retains_sweeps() -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0, 1],
            "sweep_idx": [1, 2, 3],
            "sequence_hash": ["h1", "h2", "h3"],
            "raw_llr_lexA": [0.10, 0.25, 0.60],
            "raw_llr_cpxR": [0.11, 0.30, 0.65],
        }
    )
    elites_df = pd.DataFrame(
        {
            "sequence_hash": ["h2", "h3", "hx"],
            "raw_llr_lexA": [9.0, 9.0, 9.0],
            "raw_llr_cpxR": [9.0, 9.0, 9.0],
        }
    )

    elite_rows, retain_sweeps_by_chain, stats = _prepare_elite_points(
        elites_df,
        trajectory_df=trajectory_df,
        x_col="raw_llr_lexA",
        y_col="raw_llr_cpxR",
        retain_elites=True,
    )

    assert stats["total"] == 3
    assert stats["exact_mapped"] == 2
    assert retain_sweeps_by_chain == {0: {2}, 1: {3}}
    mapped = elite_rows[elite_rows["_exact_mapped"].astype(bool)].reset_index(drop=True)
    assert mapped["raw_llr_lexA"].tolist() == [9.0, 9.0]
    assert mapped["_x"].tolist() == [0.25, 0.60]
    assert mapped["_y"].tolist() == [0.30, 0.65]


def test_sample_scatter_best_progression_projects_running_best() -> None:
    plot_df = pd.DataFrame(
        {
            "chain": [0, 0, 0, 0, 0],
            "sweep_idx": [0, 1, 2, 3, 4],
            "raw_llr_lexA": [0.0, 10.0, 20.0, 30.0, 40.0],
            "raw_llr_cpxR": [0.1, 10.1, 20.1, 30.1, 40.1],
            "objective_scalar": [1.0, 5.0, 4.0, 6.0, 3.0],
        }
    )

    sampled, best_updates = _sample_scatter_best_progression(
        plot_df,
        x_col="raw_llr_lexA",
        y_col="raw_llr_cpxR",
        objective_column="objective_scalar",
        stride=2,
        retain_sweeps_by_chain={},
    )

    assert sampled["sweep_idx"].tolist() == [0, 1, 2, 3, 4]
    assert sampled["raw_llr_lexA"].tolist() == [0.0, 10.0, 10.0, 30.0, 30.0]
    assert sampled["objective_scalar"].tolist() == [1.0, 5.0, 5.0, 6.0, 6.0]
    assert best_updates["sweep_idx"].tolist() == [0, 1, 3]


def test_sample_scatter_backbone_keeps_requested_sweep_indices() -> None:
    plot_df = pd.DataFrame(
        {
            "chain": [0, 0, 0, 0, 0],
            "sweep_idx": [0, 1, 2, 3, 4],
            "raw_llr_lexA": [0.0, 1.0, 2.0, 3.0, 4.0],
            "raw_llr_cpxR": [0.1, 1.1, 2.1, 3.1, 4.1],
            "objective_scalar": [0.0, 1.0, 0.5, 2.0, 1.5],
        }
    )

    sampled, best_updates = _sample_scatter_backbone(
        plot_df,
        x_col="raw_llr_lexA",
        y_col="raw_llr_cpxR",
        objective_column="objective_scalar",
        stride=3,
        retain_sweeps_by_chain={0: {2}},
    )

    assert sampled["sweep_idx"].tolist() == [0, 1, 2, 3, 4]
    assert best_updates["sweep_idx"].tolist() == [0, 1, 3]
