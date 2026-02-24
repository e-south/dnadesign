"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_analyze_workflow_trajectory_pair.py

Validates trajectory TF-pair resolution contracts used by analysis plotting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dnadesign.cruncher.app.analyze.score_space import (
    _resolve_score_space_spec,
    _resolve_trajectory_tf_pair,
    _resolve_worst_second_tf_pair,
)
from dnadesign.cruncher.app.analyze_workflow import (
    _summarize_elites_mmr,
)


def test_resolve_trajectory_tf_pair_uses_selected_pair() -> None:
    resolved = _resolve_trajectory_tf_pair(["lexA", "cpxR", "fur"], ["cpxR", "fur"])
    assert resolved == ("cpxR", "fur")


def test_resolve_trajectory_tf_pair_duplicates_single_tf() -> None:
    resolved = _resolve_trajectory_tf_pair(["lexA"], "auto")
    assert resolved == ("lexA", "lexA")


def test_resolve_trajectory_tf_pair_rejects_unknown_tf() -> None:
    with pytest.raises(ValueError, match="analysis.pairwise TFs must be present"):
        _resolve_trajectory_tf_pair(["lexA", "cpxR"], ["lexA", "fur"])


def test_resolve_score_space_spec_auto_uses_worst_second_for_multi_tf() -> None:
    spec = _resolve_score_space_spec(["lexA", "cpxR", "fur"], "auto")
    assert spec["mode"] == "worst_vs_second_worst"
    assert spec["x_metric"] == "worst_tf_score"
    assert spec["y_metric"] == "second_worst_tf_score"


def test_resolve_score_space_spec_all_pairs_grid_lists_all_pairs() -> None:
    spec = _resolve_score_space_spec(["lexA", "cpxR", "fur"], "all_pairs_grid")
    assert spec["mode"] == "all_pairs_grid"
    assert spec["pairs"] == [("lexA", "cpxR"), ("lexA", "fur"), ("cpxR", "fur")]


def test_resolve_score_space_spec_rejects_all_pairs_grid_for_single_tf() -> None:
    with pytest.raises(ValueError, match="requires at least two TFs"):
        _resolve_score_space_spec(["lexA"], "all_pairs_grid")


def test_resolve_worst_second_tf_pair_selects_deterministic_tf_axes() -> None:
    elites_df = pd.DataFrame(
        {
            "raw_llr_lexA": [0.10, 0.20, 0.40, 0.30],
            "raw_llr_cpxR": [0.30, 0.10, 0.20, 0.50],
            "raw_llr_fur": [0.40, 0.50, 0.10, 0.20],
        }
    )
    pair = _resolve_worst_second_tf_pair(
        elites_df=elites_df,
        tf_names=["lexA", "cpxR", "fur"],
        score_prefix="raw_llr_",
    )
    assert pair == ("fur", "cpxR")


def test_resolve_worst_second_tf_pair_requires_non_empty_elites() -> None:
    with pytest.raises(ValueError, match="without elite rows"):
        _resolve_worst_second_tf_pair(
            elites_df=pd.DataFrame(columns=["raw_llr_lexA", "raw_llr_cpxR", "raw_llr_fur"]),
            tf_names=["lexA", "cpxR", "fur"],
            score_prefix="raw_llr_",
        )


def test_summarize_elites_mmr_rejects_non_numeric_median_relevance_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dnadesign.cruncher.app.analyze_support.compute_elites_nn_distance_table",
        lambda *_args, **_kwargs: pd.DataFrame(columns=["elite_id", "identity_mode"]),
    )
    monkeypatch.setattr(
        "dnadesign.cruncher.app.analyze_support.compute_elites_full_sequence_nn_table",
        lambda *_args, **_kwargs: (pd.DataFrame(columns=["elite_id", "identity_mode"]), {}),
    )
    monkeypatch.setattr(
        "dnadesign.cruncher.app.analyze_support.representative_elite_ids",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "dnadesign.cruncher.app.analyze_support.compute_elite_distance_matrix",
        lambda *_args, **_kwargs: ([], np.zeros((0, 0))),
    )
    monkeypatch.setattr(
        "dnadesign.cruncher.app.analyze_support.summarize_elite_distances",
        lambda *_args, **_kwargs: {"mean_pairwise_distance": None, "min_pairwise_distance": None},
    )

    elites_df = pd.DataFrame(
        {
            "id": ["E1"],
            "rank": [0],
            "sequence": ["ATGC"],
            "canonical_sequence": ["ATGC"],
            "min_norm": [0.5],
        }
    )
    hits_df = pd.DataFrame({"elite_id": ["E1"]})
    sequences_df = pd.DataFrame({"phase": ["draw"], "sequence": ["ATGC"], "canonical_sequence": ["ATGC"]})
    elites_meta = {"mmr_summary": {"median_relevance_raw": "not-a-number"}}

    with pytest.raises(ValueError, match="median_relevance_raw must be numeric"):
        _summarize_elites_mmr(
            elites_df=elites_df,
            hits_df=hits_df,
            sequences_df=sequences_df,
            elites_meta=elites_meta,
            tf_names=["lexA"],
            pwms={},
            bidirectional=True,
        )
