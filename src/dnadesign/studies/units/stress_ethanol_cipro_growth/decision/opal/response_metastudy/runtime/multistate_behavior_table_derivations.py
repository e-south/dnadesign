"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_table_derivations.py

Recompute behavior summary tables from persisted lower-level evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from ..evaluation.multistate_behavior_comparison import (
    build_repeated_behavior_agreement,
    compare_hard_and_behavior_scores,
)
from ..evaluation.multistate_behavior_stability import build_bootstrap_rank_stability
from .multistate_behavior_event_verification import verify_event_score_derivations
from .multistate_behavior_frame_verification import assert_frame_equal_by_key
from .multistate_behavior_semantic_verification import BehaviorBundleSemantics


def verify_behavior_table_derivations(
    tables: dict[str, pd.DataFrame],
    *,
    semantics: BehaviorBundleSemantics,
    comparator_semantics: str,
) -> None:
    """Rebuild all cheap derived evidence from persisted lower-level tables."""

    _verify_observed_scores_from_coordinates(
        tables["observed_scores"],
        tables["observed_coordinates"],
        semantics=semantics,
    )
    verify_event_score_derivations(tables["event_sensitivity"], tables["observed_scores"])
    repeated = build_repeated_behavior_agreement(tables["observed_scores"])
    _assert_frame_equivalent(
        tables["repeated_candidate_agreement"],
        repeated,
        keys=["candidate_id", "selection_view_id"],
        context="repeated candidate agreement",
    )
    stability = build_bootstrap_rank_stability(
        tables["observed_scores"],
        tables["bootstrap_scores"],
    )
    _assert_frame_equivalent(
        tables["bootstrap_rank_draws"],
        stability.draws,
        keys=["selection_view_id", "draw_index"],
        context="bootstrap rank draws",
    )
    _assert_frame_equivalent(
        tables["bootstrap_rank_stability"],
        stability.summary,
        keys=["selection_view_id"],
        context="bootstrap rank stability",
    )
    hard = tables["hard_behavior_detail"].loc[
        :, ["id", "selection_view_id", "hard_score", "prediction_run_id", "prediction_source_sha256"]
    ]
    comparison = compare_hard_and_behavior_scores(
        hard,
        tables["prediction_scores"],
        top_k=semantics.prediction_raw_top_k,
        hard_score_semantics=comparator_semantics,
    )
    _assert_frame_equivalent(
        tables["hard_behavior_detail"],
        comparison.detail,
        keys=["selection_view_id", "id"],
        context="hard behavior detail",
    )
    _assert_frame_equivalent(
        tables["hard_behavior_summary"],
        comparison.summary,
        keys=["selection_view_id"],
        context="hard behavior summary",
    )


def _verify_observed_scores_from_coordinates(
    scores: pd.DataFrame,
    coordinates: pd.DataFrame,
    *,
    semantics: BehaviorBundleSemantics,
) -> None:
    indexed_scores = scores.set_index(["id", "selection_view_id"])
    if not indexed_scores.index.is_unique:
        raise ValueError("observed score identities must be unique before coordinate replay.")
    for (unit_id, view_id), rows in coordinates.groupby(["id", "selection_view_id"], sort=False):
        labels = _ordered_coordinate_labels(semantics.state_ids, semantics.view_masks[str(view_id)])
        indexed = rows.set_index("coordinate_label")
        if not indexed.index.is_unique:
            raise ValueError("observed coordinate labels must be unique before score replay.")
        indexed = indexed.loc[list(labels)]
        values = indexed["clearance"].to_numpy(dtype=float)
        family_counts = (
            sum(label.startswith("response:") for label in labels),
            sum(label.startswith("on_signal:") for label in labels),
            sum(label.startswith("off_signal_suppression:") for label in labels),
        )
        response, on_signal, off_signal_suppression = np.split(values, np.cumsum(family_counts)[:-1])
        family_scores = tuple(_smooth_bottleneck(family) for family in (response, on_signal, off_signal_suppression))
        prior = np.concatenate([np.full(count, 1.0 / (3.0 * count)) for count in family_counts])
        log_terms = -values + np.log(prior)
        behavior_score = float(-logsumexp(log_terms))
        weights = np.exp(log_terms - logsumexp(log_terms))
        limiting_index = int(np.argmin(values))
        expected = {
            "behavior_score": behavior_score,
            "hard_bottleneck_clearance": float(np.min(values)),
            "response_family_score": family_scores[0],
            "on_signal_family_score": family_scores[1],
            "off_signal_suppression_family_score": family_scores[2],
        }
        score_row = indexed_scores.loc[(unit_id, view_id)]
        for field, value in expected.items():
            if not np.isclose(float(score_row[field]), value, rtol=1e-12, atol=1e-12):
                raise ValueError(f"observed score {field!r} does not derive from coordinate evidence.")
        if str(score_row["limiting_coordinate"]) != labels[limiting_index]:
            raise ValueError("observed limiting coordinate does not derive from coordinate evidence.")
        if bool(score_row["all_reference_directions_met"]) != bool(np.all(values >= 0.0)):
            raise ValueError("observed reference-direction diagnostic does not derive from coordinate evidence.")
        if not np.allclose(indexed["bottleneck_weight"].to_numpy(dtype=float), weights, rtol=1e-12, atol=1e-12):
            raise ValueError("observed bottleneck weights do not derive from coordinate evidence.")
        hard_flags = indexed["is_hard_bottleneck"].to_numpy(dtype=bool)
        expected_flags = np.arange(len(values)) == limiting_index
        if not np.array_equal(hard_flags, expected_flags):
            raise ValueError("observed hard-bottleneck flags do not derive from coordinate evidence.")


def _smooth_bottleneck(values: np.ndarray) -> float:
    return float(-(logsumexp(-values) - np.log(len(values))))


def _ordered_coordinate_labels(state_ids: tuple[str, ...], mask: tuple[int, ...]) -> tuple[str, ...]:
    on = [state_ids[index] for index, value in enumerate(mask) if value == 1]
    off = [state_ids[index] for index, value in enumerate(mask) if value == 0]
    return (
        *(f"response:{left}>{right}" for left in on for right in off),
        *(f"on_signal:{state}" for state in on),
        *(f"off_signal_suppression:{state}" for state in off),
    )


def _assert_frame_equivalent(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    keys: list[str],
    context: str,
) -> None:
    try:
        assert_frame_equal_by_key(observed, expected, keys=keys)
    except AssertionError as exc:
        raise ValueError(f"{context} does not derive from its persisted source table.") from exc


__all__ = ["verify_behavior_table_derivations"]
