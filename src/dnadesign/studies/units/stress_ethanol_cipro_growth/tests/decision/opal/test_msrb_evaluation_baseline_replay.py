"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/test_msrb_evaluation_baseline_replay.py

Selection replay tests for the frozen round-0 MSRB evaluation baseline.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dnadesign.opal import score_multistate_response_behavior
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.multistate_response_behavior import (
    evaluation_baseline_artifacts,
    evaluation_baseline_contracts,
)

verify_msrb_selection_replay = evaluation_baseline_artifacts.verify_msrb_selection_replay
MsrbEvaluationBaselineError = evaluation_baseline_contracts.MsrbEvaluationBaselineError


def _campaign_config() -> dict[str, object]:
    views = (
        ("ethanol", [0, 1, 0, 1]),
        ("ciprofloxacin", [0, 0, 1, 1]),
        ("and", [0, 0, 0, 1]),
    )
    return {
        "campaign": {"slug": "toy_msrb"},
        "selection_views": [
            {
                "id": view_id,
                "objective": {
                    "name": "multistate_response_behavior_v1",
                    "params": {
                        "state_ids": ["00", "10", "01", "11"],
                        "target_mask": target_mask,
                        "softmin_scale": 0.31,
                    },
                },
                "selection": {
                    "name": "top_n",
                    "params": {
                        "top_k": 1,
                        "score_ref": "behavior_score",
                        "tie_handling": "ordinal",
                        "objective_mode": "maximize",
                        "require_exact_top_k": True,
                    },
                },
            }
            for view_id, target_mask in views
        ],
        "selection_batch": {
            "deduplicate_by": "sequence",
            "expected_unique_count": 3,
            "allocation": {
                "strategy": "round_robin_next_best_unallocated",
                "view_priority": ["ethanol", "ciprofloxacin", "and"],
            },
        },
    }


def _prediction_frame(config: dict[str, object]) -> pd.DataFrame:
    rows = (
        ("candidate-a", "AAAA", [0.0, 3.0, 0.0, 3.0, -2.0, 2.0, -2.0, 2.0]),
        ("candidate-b", "CCCC", [0.0, 0.0, 3.0, 3.0, -2.0, -2.0, 2.0, 2.0]),
        ("candidate-c", "GGGG", [0.0, 0.0, 0.0, 3.0, -2.0, -2.0, -2.0, 2.0]),
    )
    y_hat = np.asarray([row[2] for row in rows], dtype=float)
    stored_by_candidate: list[list[dict[str, object]]] = [[] for _ in rows]
    selection_views = config["selection_views"]
    assert isinstance(selection_views, list)
    for view in selection_views:
        assert isinstance(view, dict)
        objective = view["objective"]
        selection = view["selection"]
        assert isinstance(objective, dict)
        assert isinstance(selection, dict)
        params = objective["params"]
        selection_params = selection["params"]
        assert isinstance(params, dict)
        assert isinstance(selection_params, dict)
        scores = score_multistate_response_behavior(y_hat, **params).behavior_score
        for index, score in enumerate(scores):
            stored_by_candidate[index].append(
                {
                    "selection_view_id": view["id"],
                    "objective_name": objective["name"],
                    "selection_name": selection["name"],
                    "score": float(score),
                    "score_ref": f"{view['id']}/behavior_score",
                    "top_k": selection_params["top_k"],
                }
            )
    return pd.DataFrame(
        {
            "id": [row[0] for row in rows],
            "sequence": [row[1] for row in rows],
            "pred__y_hat_model": [np.asarray(row[2], dtype=float) for row in rows],
            "pred__selection_views": [np.asarray(value, dtype=object) for value in stored_by_candidate],
        }
    )


def _selection_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["candidate-a", "candidate-b", "candidate-c"],
            "selection_batch_key": ["AAAA", "CCCC", "GGGG"],
            "allocation_view_id": ["ethanol", "ciprofloxacin", "and"],
            "allocation_slot": [1, 1, 1],
        }
    )


def test_selection_replay_recomputes_scores_and_exact_allocation() -> None:
    evidence = verify_msrb_selection_replay(
        prediction_frame=_prediction_frame(_campaign_config()),
        selection_frame=_selection_frame(),
        campaign_config=_campaign_config(),
        expected_campaign_slug="toy_msrb",
        expected_allocation_api_version="1",
    )

    assert evidence.score_count == 9
    assert evidence.max_abs_score_difference == 0.0
    assert evidence.allocated_count == 3


def test_selection_replay_rejects_stored_score_or_allocation_drift() -> None:
    config = _campaign_config()
    score_drift = _prediction_frame(config)
    stored = score_drift.at[0, "pred__selection_views"].copy()
    stored[0] = {**stored[0], "score": float(stored[0]["score"]) + 0.01}
    score_drift.at[0, "pred__selection_views"] = stored
    with pytest.raises(MsrbEvaluationBaselineError, match="stored MSRB score drift"):
        verify_msrb_selection_replay(
            prediction_frame=score_drift,
            selection_frame=_selection_frame(),
            campaign_config=config,
            expected_campaign_slug="toy_msrb",
            expected_allocation_api_version="1",
        )

    allocation_drift = _selection_frame()
    allocation_drift.loc[0, "id"] = "candidate-b"
    with pytest.raises(MsrbEvaluationBaselineError, match="selection allocation drift"):
        verify_msrb_selection_replay(
            prediction_frame=_prediction_frame(config),
            selection_frame=allocation_drift,
            campaign_config=config,
            expected_campaign_slug="toy_msrb",
            expected_allocation_api_version="1",
        )
