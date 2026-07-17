"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/multistate_behavior_normalization_sensitivity.py

Prespecified scale sensitivity for the behavior shadow objective.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from dnadesign.opal import score_multistate_response_behavior

from ..core.contracts import StressTargetView
from .multistate_behavior_cohort import behavior_component_columns
from .multistate_behavior_protocol import MultistateBehaviorShadowProtocol

_RANKING_METHOD = "descending_score_then_ascending_candidate_id"
_TIE_SEMANTICS = "ordinal_rank_with_id_tiebreak"
_EVIDENCE_ROLE = "normalization_robustness_no_parameter_tuning_or_campaign_selection"


def build_multistate_behavior_normalization_sensitivity(
    *,
    response_resolution_rows: pd.DataFrame,
    signal_resolution_rows: pd.DataFrame,
    predictions: pd.DataFrame,
    protocol: MultistateBehaviorShadowProtocol,
    target_views: tuple[StressTargetView, ...],
    normalization_source_rows_sha256: str,
    prediction_run_id: str,
    prediction_source_sha256: str,
) -> pd.DataFrame:
    """Compare prespecified scales on one unchanged prediction matrix."""

    protocol.assert_target_views(target_views)
    response = _resolution_rows(response_resolution_rows, context="response")
    signal = _resolution_rows(signal_resolution_rows, context="signal")
    if set(response["id"].astype(str)) != set(signal["id"].astype(str)):
        raise ValueError("normalization sensitivity response and signal units disagree.")
    values, candidate_ids = _prediction_matrix(predictions, protocol=protocol)
    primary_scales = _scales(response, signal, quantile=protocol.completion_gate.normalization_primary_quantile)
    primary = _scores_by_view(values, target_views=target_views, protocol=protocol, scales=primary_scales)
    primary_rankings = {view_id: _ranking(candidate_ids, scores) for view_id, scores in primary.items()}
    scenarios: list[tuple[str, str, float, str, pd.DataFrame, pd.DataFrame]] = []
    for quantile in protocol.completion_gate.normalization_quantiles:
        scenarios.append(
            (f"quantile_q{int(round(quantile * 100)):02d}", "scale_quantile", quantile, "none", response, signal)
        )
    experiments = tuple(sorted(response["reader_experiment_id"].astype(str).unique()))
    if len(experiments) < 2:
        raise ValueError("normalization holdout sensitivity requires at least two source experiments.")
    for experiment_id in experiments:
        scenarios.append(
            (
                f"leave_out::{experiment_id}",
                "leave_one_source_experiment_out",
                protocol.completion_gate.normalization_primary_quantile,
                experiment_id,
                response.loc[~response["reader_experiment_id"].astype(str).eq(experiment_id)],
                signal.loc[~signal["reader_experiment_id"].astype(str).eq(experiment_id)],
            )
        )

    records: list[dict[str, object]] = []
    for scenario_id, kind, quantile, excluded, response_rows, signal_rows in scenarios:
        scales = _scales(response_rows, signal_rows, quantile=quantile)
        scenario_scores = _scores_by_view(values, target_views=target_views, protocol=protocol, scales=scales)
        for view in target_views:
            ranking = _ranking(candidate_ids, scenario_scores[view.id])
            primary_ranking = primary_rankings[view.id]
            raw_top = ranking[: protocol.prediction_raw_top_k]
            primary_top = primary_ranking[: protocol.prediction_raw_top_k]
            records.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_kind": kind,
                    "scale_quantile": float(quantile),
                    "excluded_reader_experiment_id": excluded,
                    "normalization_unit_count": int(response_rows["id"].astype(str).nunique()),
                    "selection_view_id": view.id,
                    "candidate_count": len(candidate_ids),
                    "response_scale": scales["response_scale"],
                    "signal_scale": scales["signal_scale"],
                    "score_spearman_vs_primary": _spearman(
                        scenario_scores[view.id],
                        primary[view.id],
                    ),
                    "raw_top_k": protocol.prediction_raw_top_k,
                    "raw_top_k_overlap": len(set(raw_top) & set(primary_top)),
                    "raw_top_candidate_ids_json": json.dumps(raw_top, separators=(",", ":")),
                    "primary_top_candidate_ids_json": json.dumps(primary_top, separators=(",", ":")),
                    "ranking_method": _RANKING_METHOD,
                    "tie_semantics": _TIE_SEMANTICS,
                    "objective_name": protocol.objective_name,
                    "protocol_id": protocol.protocol_id,
                    "protocol_source_sha256": f"sha256:{protocol.source_sha256}",
                    "normalization_source_rows_sha256": normalization_source_rows_sha256,
                    "prediction_run_id": prediction_run_id,
                    "prediction_source_sha256": prediction_source_sha256,
                    "evidence_role": _EVIDENCE_ROLE,
                }
            )
    return (
        pd.DataFrame.from_records(records)
        .sort_values(
            ["scenario_kind", "scenario_id", "selection_view_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _resolution_rows(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    required = {"id", "reader_experiment_id", "bootstrap_sd"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"{context} normalization sensitivity rows lack fields: {missing}")
    rows = frame.copy()
    if rows.empty or rows[["id", "reader_experiment_id"]].isna().any().any():
        raise ValueError(f"{context} normalization sensitivity rows must contain identified evidence.")
    values = rows["bootstrap_sd"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError(f"{context} normalization sensitivity SDs must be finite and nonnegative.")
    return rows


def _prediction_matrix(
    frame: pd.DataFrame,
    *,
    protocol: MultistateBehaviorShadowProtocol,
) -> tuple[np.ndarray, tuple[str, ...]]:
    components = behavior_component_columns(protocol)
    required = {"id", *components}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"normalization sensitivity predictions lack fields: {missing}")
    rows = frame.loc[:, ["id", *components]].copy()
    rows["id"] = rows["id"].astype(str)
    if rows.empty or rows["id"].str.strip().ne(rows["id"]).any() or rows["id"].duplicated().any():
        raise ValueError("normalization sensitivity candidate IDs must be exact, nonempty, and unique.")
    matrix = rows.loc[:, list(components)].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("normalization sensitivity predictions must be finite.")
    return matrix, tuple(rows["id"])


def _scales(response: pd.DataFrame, signal: pd.DataFrame, *, quantile: float) -> dict[str, float]:
    values = {
        "response_scale": float(np.quantile(response["bootstrap_sd"].to_numpy(dtype=float), quantile, method="linear")),
        "signal_scale": float(np.quantile(signal["bootstrap_sd"].to_numpy(dtype=float), quantile, method="linear")),
    }
    if not all(np.isfinite(value) and value > 0.0 for value in values.values()):
        raise ValueError("normalization sensitivity produced a nonpositive or nonfinite scale.")
    return values


def _scores_by_view(
    matrix: np.ndarray,
    *,
    target_views: tuple[StressTargetView, ...],
    protocol: MultistateBehaviorShadowProtocol,
    scales: dict[str, float],
) -> dict[str, np.ndarray]:
    return {
        view.id: score_multistate_response_behavior(
            matrix,
            state_ids=protocol.state_ids,
            target_mask=view.target_mask,
            normalization=scales,
        ).behavior_score
        for view in target_views
    }


def _ranking(candidate_ids: tuple[str, ...], scores: np.ndarray) -> list[str]:
    return [
        candidate_ids[index]
        for index in sorted(range(len(candidate_ids)), key=lambda index: (-float(scores[index]), candidate_ids[index]))
    ]


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = pd.Series(left).corr(pd.Series(right), method="spearman")
    if not np.isfinite(value):
        raise ValueError("normalization sensitivity rank correlation is undefined.")
    return float(value)


__all__ = ["build_multistate_behavior_normalization_sensitivity"]
