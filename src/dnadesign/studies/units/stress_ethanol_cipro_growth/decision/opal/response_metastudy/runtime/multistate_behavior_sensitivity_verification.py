"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_sensitivity_verification.py

Fail-closed replay of behavior normalization-sensitivity evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from dnadesign.opal import score_multistate_response_behavior

from ..evaluation.multistate_behavior_protocol import MultistateBehaviorShadowProtocol
from .multistate_behavior_semantic_verification import BehaviorBundleSemantics

_RANKING_METHOD = "descending_score_then_ascending_candidate_id"
_TIE_SEMANTICS = "ordinal_rank_with_id_tiebreak"
_EVIDENCE_ROLE = "normalization_robustness_no_parameter_tuning_or_campaign_selection"


def verify_normalization_sensitivity(
    tables: dict[str, pd.DataFrame],
    *,
    semantics: BehaviorBundleSemantics,
    protocol: MultistateBehaviorShadowProtocol,
) -> None:
    """Rebuild every declared scale, rank, and Top-K receipt from lower-level rows."""

    frame = tables["normalization_sensitivity"]
    response = tables["normalization_response_resolution"]
    signal = tables["normalization_signal_resolution"]
    vectors = tables["prediction_vectors"]
    expected_scenarios = _scenario_sources(response, signal, protocol=protocol)
    if set(frame["scenario_id"].astype(str)) != set(expected_scenarios):
        raise ValueError("normalization sensitivity scenario coverage drifted.")
    if frame.duplicated(subset=["scenario_id", "selection_view_id"]).any():
        raise ValueError("normalization sensitivity contains duplicate scenario/view rows.")
    if not frame.groupby("scenario_id")["selection_view_id"].nunique().eq(len(protocol.target_views)).all():
        raise ValueError("normalization sensitivity does not cover every view per scenario.")

    matrix = vectors.loc[:, _components(protocol)].to_numpy(dtype=float)
    candidate_ids = tuple(vectors["id"].astype(str))
    primary_scales = _scales(response, signal, protocol.completion_gate.normalization_primary_quantile)
    primary_scores = {
        view.id: _behavior_scores(matrix, view.target_mask, primary_scales, protocol=protocol)
        for view in protocol.target_views
    }
    primary_rankings = {view_id: _rank(candidate_ids, values) for view_id, values in primary_scores.items()}

    for row in frame.itertuples(index=False):
        scenario_id = str(row.scenario_id)
        kind, quantile, excluded, response_rows, signal_rows = expected_scenarios[scenario_id]
        expected_scales = _scales(response_rows, signal_rows, quantile)
        expected_literals = {
            "scenario_kind": kind,
            "excluded_reader_experiment_id": excluded,
            "normalization_unit_count": int(response_rows["id"].astype(str).nunique()),
            "candidate_count": len(candidate_ids),
            "raw_top_k": protocol.prediction_raw_top_k,
            "ranking_method": _RANKING_METHOD,
            "tie_semantics": _TIE_SEMANTICS,
            "objective_name": protocol.objective_name,
            "evidence_role": _EVIDENCE_ROLE,
        }
        for field, expected in expected_literals.items():
            if getattr(row, field) != expected:
                raise ValueError(f"normalization sensitivity {scenario_id!r} field {field!r} drifted.")
        if not np.isclose(float(row.scale_quantile), quantile, rtol=0.0, atol=0.0):
            raise ValueError("normalization sensitivity scale quantile drifted.")
        for field, expected in expected_scales.items():
            if not np.isclose(float(getattr(row, field)), expected, rtol=1e-12, atol=1e-12):
                raise ValueError(f"normalization sensitivity {field} does not derive from resolution rows.")

        view = next(item for item in protocol.target_views if item.id == str(row.selection_view_id))
        scores = _behavior_scores(matrix, view.target_mask, expected_scales, protocol=protocol)
        correlation = pd.Series(scores).corr(pd.Series(primary_scores[view.id]), method="spearman")
        if not np.isclose(float(row.score_spearman_vs_primary), float(correlation), rtol=1e-12, atol=1e-12):
            raise ValueError("normalization sensitivity rank correlation does not replay.")
        ranked = _rank(candidate_ids, scores)
        primary_ranked = primary_rankings[view.id]
        observed_ids = json.loads(str(row.raw_top_candidate_ids_json))
        observed_primary_ids = json.loads(str(row.primary_top_candidate_ids_json))
        expected_ids = ranked[: protocol.prediction_raw_top_k]
        expected_primary_ids = primary_ranked[: protocol.prediction_raw_top_k]
        if observed_ids != expected_ids or observed_primary_ids != expected_primary_ids:
            raise ValueError("normalization sensitivity Top-K identities do not replay.")
        if int(row.raw_top_k_overlap) != len(set(expected_ids) & set(expected_primary_ids)):
            raise ValueError("normalization sensitivity Top-K overlap does not replay.")
    primary = frame.loc[frame["scenario_id"].eq("quantile_q90")]
    if not np.allclose(primary["response_scale"], semantics.response_scale, rtol=1e-12, atol=0.0):
        raise ValueError("normalization sensitivity q90 response scale disagrees with the primary record.")
    if not np.allclose(primary["signal_scale"], semantics.signal_scale, rtol=1e-12, atol=0.0):
        raise ValueError("normalization sensitivity q90 signal scale disagrees with the primary record.")


def _scenario_sources(
    response: pd.DataFrame,
    signal: pd.DataFrame,
    *,
    protocol: MultistateBehaviorShadowProtocol,
) -> dict[str, tuple[str, float, str, pd.DataFrame, pd.DataFrame]]:
    scenarios = {
        f"quantile_q{int(round(value * 100)):02d}": ("scale_quantile", value, "none", response, signal)
        for value in protocol.completion_gate.normalization_quantiles
    }
    for experiment_id in sorted(response["reader_experiment_id"].astype(str).unique()):
        scenarios[f"leave_out::{experiment_id}"] = (
            "leave_one_source_experiment_out",
            protocol.completion_gate.normalization_primary_quantile,
            experiment_id,
            response.loc[~response["reader_experiment_id"].astype(str).eq(experiment_id)],
            signal.loc[~signal["reader_experiment_id"].astype(str).eq(experiment_id)],
        )
    return scenarios


def _scales(response: pd.DataFrame, signal: pd.DataFrame, quantile: float) -> dict[str, float]:
    values = {
        "response_scale": float(np.quantile(response["bootstrap_sd"].to_numpy(dtype=float), quantile, method="linear")),
        "signal_scale": float(np.quantile(signal["bootstrap_sd"].to_numpy(dtype=float), quantile, method="linear")),
    }
    if not all(np.isfinite(value) and value > 0.0 for value in values.values()):
        raise ValueError("normalization sensitivity replay produced an invalid scale.")
    return values


def _behavior_scores(
    matrix: np.ndarray,
    mask: tuple[float, ...],
    scales: dict[str, float],
    *,
    protocol: MultistateBehaviorShadowProtocol,
) -> np.ndarray:
    return score_multistate_response_behavior(
        matrix,
        state_ids=protocol.state_ids,
        target_mask=mask,
        normalization=scales,
    ).behavior_score


def _components(protocol: MultistateBehaviorShadowProtocol) -> list[str]:
    return [f"{prefix}{state}" for prefix in ("r", "b") for state in protocol.state_ids]


def _rank(ids: tuple[str, ...], values: np.ndarray) -> list[str]:
    return [ids[index] for index in sorted(range(len(ids)), key=lambda index: (-float(values[index]), ids[index]))]


__all__ = ["verify_normalization_sensitivity"]
