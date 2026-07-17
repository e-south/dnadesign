"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/multistate_behavior_shadow.py

Read-only observed, bootstrap, and prediction scoring for the shadow protocol.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dnadesign.opal import score_multistate_response_behavior

from ..core.contracts import StressTargetView
from .multistate_behavior_cohort import behavior_component_columns
from .multistate_behavior_comparison import build_repeated_behavior_agreement
from .multistate_behavior_event import build_multistate_behavior_event_sensitivity
from .multistate_behavior_normalization import (
    MultistateBehaviorNormalizationEvidence,
    verify_multistate_behavior_normalization_source,
)
from .multistate_behavior_protocol import MultistateBehaviorShadowProtocol
from .multistate_behavior_rows import bootstrap_rows_with_identity, validated_behavior_score_rows
from .multistate_behavior_stability import build_bootstrap_rank_stability


@dataclass(frozen=True)
class MultistateBehaviorShadowEvidence:
    """Observed, bootstrap, event, repeat, and predicted evidence tables."""

    observed_scores: pd.DataFrame
    observed_coordinates: pd.DataFrame
    bootstrap_scores: pd.DataFrame
    bootstrap_rank_stability: pd.DataFrame
    bootstrap_rank_draws: pd.DataFrame
    event_sensitivity: pd.DataFrame
    repeated_candidate_agreement: pd.DataFrame
    prediction_scores: pd.DataFrame
    prediction_coordinates: pd.DataFrame


def build_multistate_behavior_shadow_evidence(
    *,
    observed: pd.DataFrame,
    bootstrap_draws: pd.DataFrame,
    predictions: pd.DataFrame,
    protocol: MultistateBehaviorShadowProtocol,
    normalization: MultistateBehaviorNormalizationEvidence,
    target_views: tuple[StressTargetView, ...],
    include_prediction_coordinates: bool = False,
) -> MultistateBehaviorShadowEvidence:
    """Apply one frozen protocol to three evidence surfaces through OPAL's public API."""

    if normalization.protocol != protocol:
        raise ValueError("behavior normalization and scoring protocol identities disagree.")
    protocol.assert_target_views(target_views)
    verify_multistate_behavior_normalization_source(observed, bootstrap_draws, evidence=normalization)
    observed_rows = validated_behavior_score_rows(observed, protocol=protocol, evidence_kind="observed")
    bootstrap_rows = bootstrap_rows_with_identity(bootstrap_draws, observed_rows, protocol=protocol)
    prediction_rows = validated_behavior_score_rows(
        predictions,
        protocol=protocol,
        evidence_kind="prediction",
    )
    observed_scores, observed_coordinates = _score_rows(
        observed_rows,
        protocol=protocol,
        normalization=normalization,
        target_views=target_views,
        evidence_kind="observed",
        include_coordinates=True,
    )
    bootstrap_scores, _ = _score_rows(
        bootstrap_rows,
        protocol=protocol,
        normalization=normalization,
        target_views=target_views,
        evidence_kind="reader_joint_bootstrap",
        include_coordinates=False,
    )
    event_sensitivity = build_multistate_behavior_event_sensitivity(
        observed_rows,
        protocol=protocol,
        normalization=normalization,
        target_views=target_views,
    )
    rank_stability = build_bootstrap_rank_stability(
        observed_scores,
        bootstrap_scores,
    )
    prediction_scores, prediction_coordinates = _score_rows(
        prediction_rows,
        protocol=protocol,
        normalization=normalization,
        target_views=target_views,
        evidence_kind="fixed_prediction",
        include_coordinates=include_prediction_coordinates,
    )
    return MultistateBehaviorShadowEvidence(
        observed_scores=observed_scores,
        observed_coordinates=observed_coordinates,
        bootstrap_scores=bootstrap_scores,
        bootstrap_rank_stability=rank_stability.summary,
        bootstrap_rank_draws=rank_stability.draws,
        event_sensitivity=event_sensitivity,
        repeated_candidate_agreement=build_repeated_behavior_agreement(observed_scores),
        prediction_scores=prediction_scores,
        prediction_coordinates=prediction_coordinates,
    )


def _score_rows(
    rows: pd.DataFrame,
    *,
    protocol: MultistateBehaviorShadowProtocol,
    normalization: MultistateBehaviorNormalizationEvidence,
    target_views: tuple[StressTargetView, ...],
    evidence_kind: str,
    include_coordinates: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    component_columns = behavior_component_columns(protocol)
    matrix = rows.loc[:, list(component_columns)].to_numpy(dtype=float)
    identity_columns = [
        column
        for column in (
            "id",
            "candidate_id",
            "reader_experiment_id",
            "draw_index",
            "prediction_run_id",
            "prediction_source_sha256",
        )
        if column in rows.columns
    ]
    score_frames: list[pd.DataFrame] = []
    coordinate_frames: list[pd.DataFrame] = []
    for view in target_views:
        score = score_multistate_response_behavior(
            matrix,
            state_ids=protocol.state_ids,
            target_mask=view.target_mask,
            normalization=normalization.normalization,
        )
        frame = rows.loc[:, identity_columns].copy()
        frame["selection_view_id"] = view.id
        frame["evidence_kind"] = evidence_kind
        frame["behavior_score"] = score.behavior_score
        frame["hard_bottleneck_clearance"] = score.hard_bottleneck_clearance
        frame["response_family_score"] = score.response_family_score
        frame["on_signal_family_score"] = score.on_signal_family_score
        frame["off_signal_suppression_family_score"] = score.off_signal_suppression_family_score
        frame["limiting_coordinate"] = list(score.limiting_coordinate_label)
        frame["all_reference_directions_met"] = score.all_reference_directions_met
        _add_provenance(frame, protocol=protocol, normalization=normalization)
        score_frames.append(frame)
        if include_coordinates:
            coordinate_frames.append(
                _coordinate_rows(
                    rows.loc[:, identity_columns],
                    view_id=view.id,
                    evidence_kind=evidence_kind,
                    labels=score.coordinate_labels,
                    clearances=score.coordinate_clearances,
                    weights=score.coordinate_weights,
                    limiting_index=score.limiting_coordinate_index,
                    protocol=protocol,
                    normalization=normalization,
                )
            )
    score_rows = pd.concat(score_frames, ignore_index=True)
    coordinate_rows = pd.concat(coordinate_frames, ignore_index=True) if coordinate_frames else pd.DataFrame()
    return score_rows, coordinate_rows


def _coordinate_rows(
    identity: pd.DataFrame,
    *,
    view_id: str,
    evidence_kind: str,
    labels: tuple[str, ...],
    clearances: np.ndarray,
    weights: np.ndarray,
    limiting_index: np.ndarray,
    protocol: MultistateBehaviorShadowProtocol,
    normalization: MultistateBehaviorNormalizationEvidence,
) -> pd.DataFrame:
    row_count, coordinate_count = clearances.shape
    frame = identity.loc[identity.index.repeat(coordinate_count)].reset_index(drop=True)
    frame["selection_view_id"] = view_id
    frame["evidence_kind"] = evidence_kind
    frame["coordinate_label"] = np.tile(np.asarray(labels, dtype=object), row_count)
    frame["clearance"] = clearances.reshape(-1)
    frame["bottleneck_weight"] = weights.reshape(-1)
    coordinate_indexes = np.tile(np.arange(coordinate_count), row_count)
    frame["is_hard_bottleneck"] = coordinate_indexes == np.repeat(limiting_index, coordinate_count)
    _add_provenance(frame, protocol=protocol, normalization=normalization)
    return frame


def _add_provenance(
    frame: pd.DataFrame,
    *,
    protocol: MultistateBehaviorShadowProtocol,
    normalization: MultistateBehaviorNormalizationEvidence,
) -> None:
    frame["objective_name"] = protocol.objective_name
    frame["protocol_id"] = protocol.protocol_id
    frame["protocol_source_sha256"] = f"sha256:{protocol.source_sha256}"
    frame["normalization_source_rows_sha256"] = f"sha256:{normalization.source_rows_sha256}"
    frame["response_scale"] = normalization.response_scale
    frame["signal_scale"] = normalization.signal_scale
    frame["status"] = protocol.status
    frame["campaign_activation"] = protocol.campaign_activation
    frame["synthesis_authorization"] = protocol.synthesis_authorization


__all__ = ["MultistateBehaviorShadowEvidence", "build_multistate_behavior_shadow_evidence"]
