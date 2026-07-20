"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/multistate_behavior_grouped_validation.py

Leakage-safe grouped validation for the two response-window objectives.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from dnadesign.opal import score_multistate_response_behavior, score_response_magnitude_feasibility

from ..core.contracts import StressTargetView
from .multistate_behavior_cohort import behavior_component_columns
from .multistate_behavior_protocol import MultistateBehaviorShadowProtocol
from .response_uncertainty import build_calibration_table

_SOURCE_FIELDS = {
    "promotion_manifest_sha256",
    "candidate_records_sha256",
    "source_observation_manifest_sha256",
    "x_column_name",
}
_EVIDENCE_ROLE = "retrospective_grouped_prediction_to_truth_validation_not_prospective_hill_climb_efficacy"


def build_grouped_objective_validation(
    *,
    labels: pd.DataFrame,
    x: np.ndarray,
    response_resolution_rows: pd.DataFrame,
    signal_resolution_rows: pd.DataFrame,
    rmf_uncertainty_rows: pd.DataFrame,
    bootstrap_samples: int,
    protocol: MultistateBehaviorShadowProtocol,
    target_views: tuple[StressTargetView, ...],
    model_params: Mapping[str, object],
    source: Mapping[str, str],
) -> pd.DataFrame:
    """Fit raw Y by source-experiment fold, then score predictions and truth fairly."""

    protocol.assert_target_views(target_views)
    rows, features, y, groups = _validated_inputs(labels, x, protocol=protocol)
    source_record = _validated_source(source)
    configured_params, configured_params_sha256 = _model_params(model_params, protocol=protocol)
    group_ids = tuple(sorted(set(groups)))
    if len(group_ids) < protocol.completion_gate.validation_minimum_source_experiment_groups:
        raise ValueError(
            "grouped behavior validation has too few label-source experiments: "
            f"required>={protocol.completion_gate.validation_minimum_source_experiment_groups}, "
            f"observed={len(group_ids)}."
        )
    records: list[dict[str, object]] = []
    for seed in protocol.completion_gate.validation_seeds:
        predicted = np.empty_like(y, dtype=float)
        for heldout in group_ids:
            test = groups == heldout
            train = ~test
            fit_params = dict(configured_params)
            fit_params["random_state"] = int(seed)
            model = RandomForestRegressor(**fit_params)
            model.fit(features[train], y[train])
            predicted[test] = np.asarray(model.predict(features[test]), dtype=float).reshape(
                int(test.sum()), y.shape[1]
            )
        if not np.isfinite(predicted).all():
            raise ValueError("grouped behavior validation produced non-finite raw-Y predictions.")
        records.extend(
            _seed_score_records(
                rows,
                y=y,
                predicted=predicted,
                groups=groups,
                group_ids=group_ids,
                seed=int(seed),
                response_resolution_rows=response_resolution_rows,
                signal_resolution_rows=signal_resolution_rows,
                rmf_uncertainty_rows=rmf_uncertainty_rows,
                bootstrap_samples=bootstrap_samples,
                protocol=protocol,
                target_views=target_views,
                configured_params_sha256=configured_params_sha256,
                source=source_record,
            )
        )
    result = (
        pd.DataFrame.from_records(records)
        .sort_values(
            ["seed", "selection_view_id", "objective_name", "label_source_reader_experiment_id", "candidate_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    result["group_spearman"] = pd.array(result["group_spearman"], dtype="Float64")
    return result


def _seed_score_records(
    rows: pd.DataFrame,
    *,
    y: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    group_ids: tuple[str, ...],
    seed: int,
    response_resolution_rows: pd.DataFrame,
    signal_resolution_rows: pd.DataFrame,
    rmf_uncertainty_rows: pd.DataFrame,
    bootstrap_samples: int,
    protocol: MultistateBehaviorShadowProtocol,
    target_views: tuple[StressTargetView, ...],
    configured_params_sha256: str,
    source: dict[str, str],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for view in target_views:
        for objective_name in (protocol.objective_name, protocol.comparator_objective_name):
            observed_scores = np.empty(len(rows), dtype=float)
            predicted_scores = np.empty(len(rows), dtype=float)
            parameters_by_group: dict[str, dict[str, float | str]] = {}
            group_correlations: dict[str, float | None] = {}
            for heldout in group_ids:
                selected = groups == heldout
                heldout_candidate_ids = frozenset(rows.loc[selected, "candidate_id"].astype(str))
                excluded_unit_ids = frozenset(
                    response_resolution_rows.loc[
                        response_resolution_rows["candidate_id"].astype(str).isin(heldout_candidate_ids),
                        "id",
                    ].astype(str)
                )
                parameters = _fold_parameters(
                    objective_name=objective_name,
                    heldout=heldout,
                    heldout_candidate_ids=heldout_candidate_ids,
                    excluded_unit_ids=excluded_unit_ids,
                    view=view,
                    protocol=protocol,
                    response_resolution_rows=response_resolution_rows,
                    signal_resolution_rows=signal_resolution_rows,
                    rmf_uncertainty_rows=rmf_uncertainty_rows,
                    bootstrap_samples=bootstrap_samples,
                )
                parameters_by_group[heldout] = parameters
                observed_scores[selected] = _objective_scores(
                    y[selected], objective_name=objective_name, view=view, protocol=protocol, parameters=parameters
                )
                predicted_scores[selected] = _objective_scores(
                    predicted[selected],
                    objective_name=objective_name,
                    view=view,
                    protocol=protocol,
                    parameters=parameters,
                )
                group_correlations[heldout] = _optional_spearman(
                    observed_scores[selected],
                    predicted_scores[selected],
                )
            defined = [value for value in group_correlations.values() if value is not None]
            if not defined:
                raise ValueError(f"grouped validation has no rank-defined folds for {view.id}/{objective_name}.")
            pooled = _optional_spearman(observed_scores, predicted_scores)
            if pooled is None:
                raise ValueError(f"grouped validation pooled ordering is undefined for {view.id}/{objective_name}.")
            median_within = float(np.median(np.asarray(defined, dtype=float)))
            for index, row in enumerate(rows.itertuples(index=False)):
                heldout = str(row.label_source_reader_experiment_id)
                parameters_json = json.dumps(
                    parameters_by_group[heldout], allow_nan=False, separators=(",", ":"), sort_keys=True
                )
                output.append(
                    {
                        "candidate_id": str(row.candidate_id),
                        "display_label": str(row.display_label),
                        "label_source_reader_experiment_id": heldout,
                        "seed": seed,
                        "selection_view_id": view.id,
                        "objective_name": objective_name,
                        "observed_y": y[index].astype(float).tolist(),
                        "predicted_y": predicted[index].astype(float).tolist(),
                        "observed_score": float(observed_scores[index]),
                        "predicted_score": float(predicted_scores[index]),
                        "heldout_candidate_count": int(np.sum(groups == heldout)),
                        "group_spearman": group_correlations[heldout],
                        "group_spearman_defined": group_correlations[heldout] is not None,
                        "rank_defined_group_count": len(defined),
                        "median_within_group_spearman": median_within,
                        "pooled_oof_spearman": pooled,
                        "normalization_parameters_json": parameters_json,
                        "normalization_parameters_sha256": "sha256:"
                        + hashlib.sha256(parameters_json.encode("utf-8")).hexdigest(),
                        "split_strategy": protocol.completion_gate.validation_split,
                        "x_preprocessing": protocol.completion_gate.validation_x_preprocessing,
                        "y_fit_space": protocol.completion_gate.validation_y_fit_space,
                        "scoring_parameter_scope": protocol.completion_gate.validation_scoring_parameters,
                        "primary_validation_metric": protocol.completion_gate.validation_primary_metric,
                        "secondary_validation_metric": protocol.completion_gate.validation_secondary_metric,
                        "model_name": "random_forest",
                        "configured_model_params_sha256": configured_params_sha256,
                        "promoted_label_count": len(rows),
                        "label_source_contract": "verified_observed_label_promotion_exact_only",
                        **source,
                        "protocol_id": protocol.protocol_id,
                        "protocol_source_sha256": f"sha256:{protocol.source_sha256}",
                        "evidence_role": _EVIDENCE_ROLE,
                    }
                )
    return output


def _fold_parameters(
    *,
    objective_name: str,
    heldout: str,
    heldout_candidate_ids: frozenset[str],
    excluded_unit_ids: frozenset[str],
    view: StressTargetView,
    protocol: MultistateBehaviorShadowProtocol,
    response_resolution_rows: pd.DataFrame,
    signal_resolution_rows: pd.DataFrame,
    rmf_uncertainty_rows: pd.DataFrame,
    bootstrap_samples: int,
) -> dict[str, float | str]:
    quantile = protocol.completion_gate.normalization_primary_quantile
    candidate_digest = _candidate_set_sha256(heldout_candidate_ids)
    exclusion = {
        "excluded_candidate_count": len(heldout_candidate_ids),
        "excluded_candidate_ids_sha256": candidate_digest,
        "excluded_normalization_unit_count": len(excluded_unit_ids),
        "excluded_source_experiment": heldout,
    }
    if objective_name == protocol.objective_name:
        response = _training_resolution_values(
            response_resolution_rows,
            heldout=heldout,
            heldout_candidate_ids=heldout_candidate_ids,
        )
        signal = _training_resolution_values(
            signal_resolution_rows,
            heldout=heldout,
            heldout_candidate_ids=heldout_candidate_ids,
        )
        return {
            **exclusion,
            "softmin_scale": _positive_quantile(np.concatenate([response, signal]), quantile),
            "scale_basis": protocol.normalization.scale_basis,
            "scale_quantile": quantile,
        }
    if objective_name != protocol.comparator_objective_name:
        raise ValueError(f"unknown grouped validation objective {objective_name!r}.")
    rmf_training_rows = rmf_uncertainty_rows.loc[
        ~rmf_uncertainty_rows["reader_experiment_id"].astype(str).eq(heldout)
        & ~rmf_uncertainty_rows["id"].astype(str).isin(excluded_unit_ids)
    ]
    table = build_calibration_table(
        rmf_training_rows,
        scale_quantile=quantile,
        bootstrap_samples=bootstrap_samples,
    )
    view_rows = table.loc[table["selection_view_id"].astype(str).eq(view.id)].set_index("component")
    return {
        **exclusion,
        "off_magnitude_max": 0.0,
        "off_magnitude_scale": float(view_rows.loc["off_magnitude_ceiling", "scale"]),
        "on_magnitude_min": 0.0,
        "on_magnitude_scale": float(view_rows.loc["on_magnitude_floor", "scale"]),
        "response_separation_min": 0.0,
        "response_separation_scale": float(view_rows.loc["response_separation", "scale"]),
        "scale_basis": "reader_joint_bootstrap_plus_conservative_event_bound",
        "scale_quantile": quantile,
    }


def _training_resolution_values(
    frame: pd.DataFrame,
    *,
    heldout: str,
    heldout_candidate_ids: frozenset[str],
) -> np.ndarray:
    required = {"candidate_id", "reader_experiment_id", "bootstrap_sd"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"fold-local normalization evidence lacks fields: {missing}")
    selected = ~frame["reader_experiment_id"].astype(str).eq(heldout) & ~frame["candidate_id"].astype(str).isin(
        heldout_candidate_ids
    )
    return frame.loc[selected, "bootstrap_sd"].to_numpy(dtype=float)


def _candidate_set_sha256(candidate_ids: frozenset[str]) -> str:
    canonical = json.dumps(sorted(candidate_ids), ensure_ascii=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _objective_scores(
    values: np.ndarray,
    *,
    objective_name: str,
    view: StressTargetView,
    protocol: MultistateBehaviorShadowProtocol,
    parameters: Mapping[str, float | str],
) -> np.ndarray:
    if objective_name == protocol.objective_name:
        return score_multistate_response_behavior(
            values,
            state_ids=protocol.state_ids,
            target_mask=view.target_mask,
            softmin_scale=float(parameters["softmin_scale"]),
        ).behavior_score
    calibration = {
        field: float(parameters[field])
        for field in (
            "response_separation_min",
            "on_magnitude_min",
            "off_magnitude_max",
            "response_separation_scale",
            "on_magnitude_scale",
            "off_magnitude_scale",
        )
    }
    return score_response_magnitude_feasibility(
        values,
        target_mask=view.target_mask,
        calibration=calibration,
    ).feasibility_margin


def _validated_inputs(
    labels: pd.DataFrame,
    x: np.ndarray,
    *,
    protocol: MultistateBehaviorShadowProtocol,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    components = behavior_component_columns(protocol)
    required = {"candidate_id", "display_label", "label_source_reader_experiment_id", *components}
    if missing := sorted(required - set(labels.columns)):
        raise ValueError(f"grouped behavior validation labels lack fields: {missing}")
    rows = labels.loc[:, ["candidate_id", "display_label", "label_source_reader_experiment_id", *components]].copy()
    if rows.empty or rows["candidate_id"].astype(str).duplicated().any():
        raise ValueError("grouped behavior validation requires unique promoted candidate labels.")
    for field in ("candidate_id", "display_label", "label_source_reader_experiment_id"):
        values = rows[field].astype(str)
        if values.str.strip().ne(values).any() or values.eq("").any():
            raise ValueError(f"grouped behavior validation {field} values must be exact and nonempty.")
        rows[field] = values
    y = rows.loc[:, list(components)].to_numpy(dtype=float)
    features = np.asarray(x, dtype=float)
    if features.ndim != 2 or len(features) != len(rows) or not np.isfinite(features).all() or not np.isfinite(y).all():
        raise ValueError("grouped behavior validation requires aligned finite two-dimensional X and raw Y.")
    return rows.reset_index(drop=True), features, y, rows["label_source_reader_experiment_id"].to_numpy(dtype=str)


def _validated_source(source: Mapping[str, str]) -> dict[str, str]:
    if set(source) != _SOURCE_FIELDS:
        raise ValueError("grouped behavior validation source fields are incomplete or unexpected.")
    result = {field: str(source[field]) for field in sorted(source)}
    for field in _SOURCE_FIELDS - {"x_column_name"}:
        value = result[field]
        if not value.startswith("sha256:") or len(value) != 71:
            raise ValueError(f"grouped behavior validation {field} must be a canonical SHA-256 digest.")
    if not result["x_column_name"] or result["x_column_name"].strip() != result["x_column_name"]:
        raise ValueError("grouped behavior validation x_column_name must be exact and nonempty.")
    return result


def _model_params(
    model_params: Mapping[str, object],
    *,
    protocol: MultistateBehaviorShadowProtocol,
) -> tuple[dict[str, object], str]:
    configured = dict(model_params)
    emit_feature_importance = configured.pop("emit_feature_importance", False)
    expanded = RandomForestRegressor(**configured).get_params(deep=False)
    expanded.pop("random_state", None)
    expanded["emit_feature_importance"] = emit_feature_importance
    canonical = json.dumps(expanded, allow_nan=False, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if digest != protocol.completion_gate.validation_model_nonseed_params_sha256:
        raise ValueError(
            "grouped behavior validation model parameters drifted from the registered prediction-run contract."
        )
    expanded.pop("emit_feature_importance")
    return expanded, "sha256:" + digest


def _positive_quantile(values: np.ndarray, quantile: float) -> float:
    if values.size == 0 or not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("fold-local normalization evidence must be finite, nonnegative, and nonempty.")
    result = float(np.quantile(values, quantile, method="linear"))
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("fold-local normalization scale must be positive and finite.")
    return result


def _optional_spearman(observed: np.ndarray, predicted: np.ndarray) -> float | None:
    if len(observed) < 2 or len(np.unique(observed)) < 2 or len(np.unique(predicted)) < 2:
        return None
    value = pd.Series(observed).corr(pd.Series(predicted), method="spearman")
    return float(value) if np.isfinite(value) else None


__all__ = ["build_grouped_objective_validation"]
