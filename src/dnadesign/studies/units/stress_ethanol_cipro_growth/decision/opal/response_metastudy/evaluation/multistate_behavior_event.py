"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/multistate_behavior_event.py

Conservative event-time envelopes for multistate behavior evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dnadesign.opal import score_multistate_response_behavior

from ..core.contracts import StressTargetView
from .multistate_behavior_normalization import MultistateBehaviorNormalizationEvidence
from .multistate_behavior_protocol import MultistateBehaviorShadowProtocol


def build_multistate_behavior_event_sensitivity(
    observed: pd.DataFrame,
    *,
    protocol: MultistateBehaviorShadowProtocol,
    normalization: MultistateBehaviorNormalizationEvidence,
    target_views: tuple[StressTargetView, ...],
) -> pd.DataFrame:
    """Score componentwise worst/best event envelopes without a probability claim."""

    components = _component_columns(protocol)
    half_range_columns = [f"{column}_event_half_range" for column in components]
    clipping_columns = [f"{column}_event_sensitivity_has_policy_clipping" for column in components]
    overflow_columns = [f"{column}_event_sensitivity_has_instrument_overflow" for column in components]
    required = {*half_range_columns, *clipping_columns, *overflow_columns}
    missing = sorted(required - set(observed.columns))
    if missing:
        raise ValueError(f"observed event-sensitivity rows missing columns: {missing}")
    half_ranges = observed.loc[:, half_range_columns].to_numpy(dtype=float)
    if not np.isfinite(half_ranges).all() or np.any(half_ranges < 0.0):
        raise ValueError("observed event half-ranges must be finite and nonnegative for all eight components.")
    clipping = _strict_boolean_flags(observed, columns=clipping_columns)
    overflow = _strict_boolean_flags(observed, columns=overflow_columns)
    if clipping.any(axis=None):
        raise ValueError("behavior event envelopes require event-sensitivity values without policy clipping.")
    if overflow.any(axis=None):
        raise ValueError("behavior event envelopes require event-sensitivity values without instrument overflow.")
    central = observed.loc[:, components].to_numpy(dtype=float)
    frames: list[pd.DataFrame] = []
    for view in target_views:
        on = np.asarray(view.target_mask, dtype=bool)
        desirable_sign = np.concatenate((np.where(on, 1.0, -1.0), np.where(on, 1.0, -1.0)))
        worst = central - half_ranges * desirable_sign[None, :]
        best = central + half_ranges * desirable_sign[None, :]
        central_score = _score(central, view=view, protocol=protocol, normalization=normalization)
        worst_score = _score(worst, view=view, protocol=protocol, normalization=normalization)
        best_score = _score(best, view=view, protocol=protocol, normalization=normalization)
        frame = observed.loc[:, ["id", "candidate_id", "reader_experiment_id"]].copy()
        frame["selection_view_id"] = view.id
        frame["behavior_score_central"] = central_score.behavior_score
        frame["behavior_score_worst_envelope"] = worst_score.behavior_score
        frame["behavior_score_best_envelope"] = best_score.behavior_score
        frame["behavior_score_envelope_width"] = (
            frame["behavior_score_best_envelope"] - frame["behavior_score_worst_envelope"]
        )
        frame["hard_bottleneck_worst_envelope"] = worst_score.hard_bottleneck_clearance
        frame["hard_bottleneck_best_envelope"] = best_score.hard_bottleneck_clearance
        frame = _rank_event_columns(frame)
        frame["event_bound_semantics"] = "componentwise_conservative_not_joint_event_draw"
        frame["event_bound_probability_claim"] = "none"
        frame["event_censor_posture"] = "exact_unclipped_unoverflowed"
        _add_provenance(frame, protocol=protocol, normalization=normalization)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _score(
    values: np.ndarray,
    *,
    view: StressTargetView,
    protocol: MultistateBehaviorShadowProtocol,
    normalization: MultistateBehaviorNormalizationEvidence,
):
    return score_multistate_response_behavior(
        values,
        state_ids=protocol.state_ids,
        target_mask=view.target_mask,
        softmin_scale=normalization.softmin_scale,
    )


def _rank_event_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for score_column, rank_column in (
        ("behavior_score_central", "central_unit_rank"),
        ("behavior_score_worst_envelope", "worst_envelope_unit_rank"),
        ("behavior_score_best_envelope", "best_envelope_unit_rank"),
    ):
        ranked = result.sort_values([score_column, "id"], ascending=[False, True], kind="mergesort").copy()
        ranked[rank_column] = np.arange(1, len(ranked) + 1, dtype=int)
        result[rank_column] = result["id"].astype(str).map(ranked.set_index("id")[rank_column]).astype(int)
    rank_columns = ["central_unit_rank", "worst_envelope_unit_rank", "best_envelope_unit_rank"]
    result["event_unit_rank_min"] = result[rank_columns].min(axis=1).astype(int)
    result["event_unit_rank_max"] = result[rank_columns].max(axis=1).astype(int)
    result["event_unit_rank_span"] = result["event_unit_rank_max"] - result["event_unit_rank_min"]
    result["ranking_method"] = "descending_score_then_ascending_candidate_experiment_unit_id"
    result["tie_semantics"] = "ordinal_rank_with_id_tiebreak"
    return result


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
    frame["softmin_scale"] = normalization.softmin_scale
    frame["status"] = protocol.status
    frame["campaign_activation"] = protocol.campaign_activation
    frame["synthesis_authorization"] = protocol.synthesis_authorization


def _component_columns(protocol: MultistateBehaviorShadowProtocol) -> list[str]:
    return [f"r{state}" for state in protocol.state_ids] + [f"b{state}" for state in protocol.state_ids]


def _strict_boolean_flags(frame: pd.DataFrame, *, columns: list[str]) -> pd.DataFrame:
    values = frame.loc[:, columns]
    invalid = values.map(lambda value: not isinstance(value, (bool, np.bool_)))
    if invalid.any(axis=None):
        raise ValueError("behavior event censor flags must contain exact boolean values, not text or truthy aliases.")
    return values.astype(bool)


__all__ = ["build_multistate_behavior_event_sensitivity"]
