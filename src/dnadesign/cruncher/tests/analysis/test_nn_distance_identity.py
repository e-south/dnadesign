"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_nn_distance_identity.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dnadesign.cruncher.analysis.diversity import (
    compute_elite_distance_matrix,
    compute_elites_full_sequence_nn_table,
    compute_elites_nn_distance_table,
)
from dnadesign.cruncher.analysis.plots.elites_nn_distance import _full_nn_xlabel, _resolve_joint_score
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.selection.mmr import compute_core_distance, compute_position_weights


def _pwm(name: str, length: int = 4) -> PWM:
    matrix = np.full((length, 4), 0.25)
    return PWM(name=name, matrix=matrix)


def test_nn_distance_ignores_duplicate_identities() -> None:
    hits_df = pd.DataFrame(
        {
            "elite_id": ["e1", "e2", "e3", "e4"],
            "tf": ["tf1", "tf1", "tf1", "tf1"],
            "best_core_seq": ["AAAA", "AAAA", "CCCC", "GGGG"],
        }
    )
    pwms = {"tf1": _pwm("tf1")}
    identity_by_elite_id = {
        "e1": "canonA",
        "e2": "canonA",
        "e3": "canonC",
        "e4": "canonG",
    }
    rank_by_elite_id = {"e1": 1, "e2": 2, "e3": 3, "e4": 4}

    nn_df = compute_elites_nn_distance_table(
        hits_df,
        ["tf1"],
        pwms,
        identity_mode="canonical",
        identity_by_elite_id=identity_by_elite_id,
        rank_by_elite_id=rank_by_elite_id,
    )

    assert len(nn_df) == 4
    row_e1 = nn_df[nn_df["elite_id"] == "e1"].iloc[0]
    row_e2 = nn_df[nn_df["elite_id"] == "e2"].iloc[0]
    assert row_e1["nn_dist"] == row_e2["nn_dist"]


def test_analysis_core_distance_matches_selection_metric_weights() -> None:
    hits_df = pd.DataFrame(
        {
            "elite_id": ["e1", "e2"],
            "tf": ["tf1", "tf1"],
            "best_core_seq": ["AACG", "AATG"],
        }
    )
    pwm = PWM(
        name="tf1",
        matrix=np.asarray(
            [
                [0.90, 0.03, 0.03, 0.04],
                [0.25, 0.25, 0.25, 0.25],
                [0.70, 0.10, 0.10, 0.10],
                [0.25, 0.25, 0.25, 0.25],
            ],
            dtype=float,
        ),
    )
    pwms = {"tf1": pwm}
    elite_ids, dist = compute_elite_distance_matrix(hits_df, ["tf1"], pwms)
    assert elite_ids == ["e1", "e2"]
    observed = float(dist[0, 1])

    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    cores_a = {"tf1": np.asarray([mapping[ch] for ch in "AACG"], dtype=np.int8)}
    cores_b = {"tf1": np.asarray([mapping[ch] for ch in "AATG"], dtype=np.int8)}
    weights = {"tf1": compute_position_weights(pwm)}
    expected = compute_core_distance(cores_a, cores_b, weights=weights, tf_names=["tf1"])

    assert observed == expected


def test_full_sequence_nn_table_reports_bp_metrics() -> None:
    elites_df = pd.DataFrame(
        {
            "id": ["e1", "e2", "e3"],
            "sequence": ["AAAA", "AAAT", "TTTT"],
            "rank": [1, 2, 3],
        }
    )
    table, summary = compute_elites_full_sequence_nn_table(
        elites_df,
        identity_mode="canonical",
        identity_by_elite_id={"e1": "canon1", "e2": "canon2", "e3": "canon3"},
        rank_by_elite_id={"e1": 1, "e2": 2, "e3": 3},
    )
    assert set(table.columns) == {
        "elite_id",
        "nn_full_bp",
        "mean_full_bp",
        "min_full_bp",
        "nn_full_dist",
        "mean_full_dist",
        "min_full_dist",
        "identity_mode",
    }
    row_e1 = table[table["elite_id"] == "e1"].iloc[0]
    assert float(row_e1["nn_full_bp"]) == 1.0
    assert float(row_e1["nn_full_dist"]) == 0.25
    assert summary["sequence_length_bp"] == 4
    assert summary["min_pairwise_full_bp"] == 1.0
    assert summary["min_pairwise_full_distance"] == 0.25


def test_resolve_joint_score_fallback_respects_sum_combine() -> None:
    elites_df = pd.DataFrame(
        {
            "score_tfA": [1.0, 2.0],
            "score_tfB": [3.0, 4.0],
        }
    )
    score, source = _resolve_joint_score(
        elites_df,
        objective_config={"combine": "sum", "softmin": {"enabled": False}},
    )
    assert source == "reconstructed_sum"
    assert score.tolist() == [4.0, 6.0]


def test_full_sequence_nn_axis_label_mentions_closest_selected_elite() -> None:
    label = _full_nn_xlabel()
    assert "closest selected elite" in label
    assert "Hamming" in label
