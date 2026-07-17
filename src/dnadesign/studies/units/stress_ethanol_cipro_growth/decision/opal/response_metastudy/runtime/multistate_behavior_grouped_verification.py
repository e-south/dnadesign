"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_grouped_verification.py

Fail-closed replay of grouped prediction-to-truth evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from dnadesign.opal import score_multistate_response_behavior, score_response_magnitude_feasibility

from ..evaluation.multistate_behavior_protocol import MultistateBehaviorShadowProtocol

_EVIDENCE_ROLE = "retrospective_grouped_prediction_to_truth_validation_not_prospective_hill_climb_efficacy"


def verify_grouped_objective_validation(
    tables: dict[str, pd.DataFrame],
    *,
    protocol: MultistateBehaviorShadowProtocol,
) -> None:
    """Replay fold exclusions, objective scores, and grouped rank evidence."""

    frame = tables["grouped_objective_validation"]
    keys = ["candidate_id", "seed", "selection_view_id", "objective_name"]
    if frame.empty or frame.duplicated(subset=keys).any():
        raise ValueError("grouped objective validation requires one row per candidate/seed/view/objective.")
    candidates = _candidate_contract(frame)
    expected_rows = len(candidates) * len(protocol.completion_gate.validation_seeds) * len(protocol.target_views) * 2
    if len(frame) != expected_rows:
        raise ValueError("grouped objective validation product coverage drifted.")
    if set(frame["seed"].astype(int)) != set(protocol.completion_gate.validation_seeds):
        raise ValueError("grouped objective validation seed coverage drifted.")
    if set(frame["objective_name"].astype(str)) != {protocol.objective_name, protocol.comparator_objective_name}:
        raise ValueError("grouped objective validation objective coverage drifted.")
    if set(frame["selection_view_id"].astype(str)) != {view.id for view in protocol.target_views}:
        raise ValueError("grouped objective validation view coverage drifted.")
    if not frame["promoted_label_count"].eq(len(candidates)).all():
        raise ValueError("grouped objective validation promoted-label count drifted.")
    _verify_source_contract(frame, protocol=protocol)
    _verify_raw_vector_consistency(frame)
    _verify_rows_and_parameters(tables, candidates=candidates, protocol=protocol)
    _verify_correlations(frame)


def _candidate_contract(frame: pd.DataFrame) -> pd.DataFrame:
    identities = frame.loc[
        :,
        ["candidate_id", "display_label", "label_source_reader_experiment_id", "observed_y"],
    ].copy()
    identities["observed_y_json"] = identities["observed_y"].map(_canonical_vector_json)
    for field in ("candidate_id", "display_label", "label_source_reader_experiment_id"):
        values = identities[field]
        if values.isna().any():
            raise ValueError(f"grouped objective validation candidate {field} must be non-null.")
        strings = values.astype(str)
        if strings.eq("").any() or strings.str.strip().ne(strings).any():
            raise ValueError(f"grouped objective validation candidate {field} must be exact and nonempty.")
    for field in ("display_label", "label_source_reader_experiment_id", "observed_y_json"):
        if not identities.groupby("candidate_id")[field].nunique().eq(1).all():
            raise ValueError(f"grouped objective validation candidate {field} drifts across rows.")
    return identities.drop_duplicates("candidate_id").reset_index(drop=True)


def _verify_source_contract(frame: pd.DataFrame, *, protocol: MultistateBehaviorShadowProtocol) -> None:
    expected = {
        "split_strategy": protocol.completion_gate.validation_split,
        "x_preprocessing": protocol.completion_gate.validation_x_preprocessing,
        "y_fit_space": protocol.completion_gate.validation_y_fit_space,
        "scoring_parameter_scope": protocol.completion_gate.validation_scoring_parameters,
        "primary_validation_metric": protocol.completion_gate.validation_primary_metric,
        "secondary_validation_metric": protocol.completion_gate.validation_secondary_metric,
        "model_name": protocol.completion_gate.validation_model_name,
        "label_source_contract": "verified_observed_label_promotion_exact_only",
        "evidence_role": _EVIDENCE_ROLE,
    }
    for field, value in expected.items():
        if set(frame[field].astype(str)) != {str(value)}:
            raise ValueError(f"grouped validation contract field {field!r} drifted.")
    for field in (
        "candidate_records_sha256",
        "promotion_manifest_sha256",
        "source_observation_manifest_sha256",
    ):
        values = frame[field].astype(str)
        if values.nunique() != 1 or not values.str.fullmatch(r"sha256:[0-9a-f]{64}").all():
            raise ValueError(f"grouped validation source digest {field!r} drifted.")
    x_column = frame["x_column_name"].astype(str)
    if x_column.nunique() != 1 or x_column.str.strip().ne(x_column).any() or x_column.eq("").any():
        raise ValueError("grouped validation X column identity drifted.")


def _verify_raw_vector_consistency(frame: pd.DataFrame) -> None:
    rows = frame.copy()
    rows["observed_y_json"] = rows["observed_y"].map(_canonical_vector_json)
    rows["predicted_y_json"] = rows["predicted_y"].map(_canonical_vector_json)
    if not rows.groupby("candidate_id")["observed_y_json"].nunique().eq(1).all():
        raise ValueError("grouped observed raw Y drifts across seeds, views, or objectives.")
    if not rows.groupby(["candidate_id", "seed"])["predicted_y_json"].nunique().eq(1).all():
        raise ValueError("grouped predicted raw Y drifts across views or objectives.")


def _verify_rows_and_parameters(
    tables: dict[str, pd.DataFrame],
    *,
    candidates: pd.DataFrame,
    protocol: MultistateBehaviorShadowProtocol,
) -> None:
    frame = tables["grouped_objective_validation"]
    response = tables["normalization_response_resolution"]
    signal = tables["normalization_signal_resolution"]
    rmf = tables["grouped_rmf_resolution"]
    group_map = candidates.set_index("candidate_id")["label_source_reader_experiment_id"].astype(str)
    cache: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in frame.itertuples(index=False):
        heldout = str(row.label_source_reader_experiment_id)
        cache_key = (str(row.objective_name), str(row.selection_view_id), heldout)
        parameters = cache.setdefault(
            cache_key,
            _fold_parameters(
                objective_name=cache_key[0],
                view_id=cache_key[1],
                heldout=heldout,
                group_map=group_map,
                response=response,
                signal=signal,
                rmf=rmf,
                protocol=protocol,
            ),
        )
        parameters_json = json.dumps(parameters, allow_nan=False, separators=(",", ":"), sort_keys=True)
        if str(row.normalization_parameters_json) != parameters_json:
            raise ValueError("grouped objective validation fold parameters do not derive from training-only rows.")
        digest = "sha256:" + hashlib.sha256(parameters_json.encode("utf-8")).hexdigest()
        if str(row.normalization_parameters_sha256) != digest:
            raise ValueError("grouped objective validation parameter digest drifted.")
        observed_score = _objective_score(_vector(row.observed_y), cache_key[0], parameters, cache_key[1], protocol)
        predicted_score = _objective_score(_vector(row.predicted_y), cache_key[0], parameters, cache_key[1], protocol)
        if not np.isclose(float(row.observed_score), observed_score, rtol=1e-12, atol=1e-12):
            raise ValueError("grouped observed objective score does not replay.")
        if not np.isclose(float(row.predicted_score), predicted_score, rtol=1e-12, atol=1e-12):
            raise ValueError("grouped predicted objective score does not replay.")
        expected_count = int(group_map.eq(heldout).sum())
        if int(row.heldout_candidate_count) != expected_count:
            raise ValueError("grouped heldout-candidate count drifted.")


def _fold_parameters(
    *,
    objective_name: str,
    view_id: str,
    heldout: str,
    group_map: pd.Series,
    response: pd.DataFrame,
    signal: pd.DataFrame,
    rmf: pd.DataFrame,
    protocol: MultistateBehaviorShadowProtocol,
) -> dict[str, object]:
    candidates = frozenset(group_map.index[group_map.eq(heldout)].astype(str))
    excluded_units = frozenset(response.loc[response["candidate_id"].astype(str).isin(candidates), "id"].astype(str))
    exclusion: dict[str, object] = {
        "excluded_candidate_count": len(candidates),
        "excluded_candidate_ids_sha256": _candidate_set_sha256(candidates),
        "excluded_normalization_unit_count": len(excluded_units),
        "excluded_source_experiment": heldout,
    }
    quantile = protocol.completion_gate.normalization_primary_quantile
    if objective_name == protocol.objective_name:
        return {
            **exclusion,
            "response_scale": _resolution_quantile(response, heldout, candidates, quantile),
            "scale_basis": "reader_joint_bootstrap_component_resolution",
            "scale_quantile": quantile,
            "signal_scale": _resolution_quantile(signal, heldout, candidates, quantile),
        }
    if objective_name != protocol.comparator_objective_name:
        raise ValueError(f"grouped objective {objective_name!r} is unknown.")
    training = rmf.loc[
        ~rmf["reader_experiment_id"].astype(str).eq(heldout) & ~rmf["id"].astype(str).isin(excluded_units)
    ]
    view_rows = training.loc[training["selection_view_id"].astype(str).eq(view_id)]
    scales = {
        component: _positive_quantile(view_rows[f"{component}__combined_sd"], quantile)
        for component in ("response_separation", "on_magnitude_floor", "off_magnitude_ceiling")
    }
    return {
        **exclusion,
        "off_magnitude_max": 0.0,
        "off_magnitude_scale": scales["off_magnitude_ceiling"],
        "on_magnitude_min": 0.0,
        "on_magnitude_scale": scales["on_magnitude_floor"],
        "response_separation_min": 0.0,
        "response_separation_scale": scales["response_separation"],
        "scale_basis": "reader_joint_bootstrap_plus_conservative_event_bound",
        "scale_quantile": quantile,
    }


def _resolution_quantile(frame: pd.DataFrame, heldout: str, candidates: frozenset[str], quantile: float) -> float:
    training = frame.loc[
        ~frame["reader_experiment_id"].astype(str).eq(heldout) & ~frame["candidate_id"].astype(str).isin(candidates),
        "bootstrap_sd",
    ]
    return _positive_quantile(training, quantile)


def _verify_correlations(frame: pd.DataFrame) -> None:
    for _, rows in frame.groupby(["seed", "selection_view_id", "objective_name"], sort=False):
        correlations: list[float] = []
        for _, group in rows.groupby("label_source_reader_experiment_id", sort=False):
            correlation = _optional_spearman(group["observed_score"], group["predicted_score"])
            defined = correlation is not None
            stored = group["group_spearman"]
            if not group["group_spearman_defined"].eq(defined).all():
                raise ValueError("grouped correlation-defined flags drifted.")
            if defined:
                if not np.allclose(stored.astype(float), correlation, rtol=1e-12, atol=1e-12):
                    raise ValueError("grouped within-heldout correlation does not replay.")
                correlations.append(float(correlation))
            elif not stored.isna().all():
                raise ValueError("grouped undefined correlations must remain null.")
        median = float(np.median(correlations))
        pooled = _optional_spearman(rows["observed_score"], rows["predicted_score"])
        if pooled is None:
            raise ValueError("grouped pooled out-of-fold correlation is undefined.")
        expected_count = len(correlations)
        if not rows["rank_defined_group_count"].eq(expected_count).all():
            raise ValueError("grouped rank-defined support count drifted.")
        if not np.allclose(rows["median_within_group_spearman"], median, rtol=1e-12, atol=1e-12):
            raise ValueError("grouped primary metric does not replay.")
        if not np.allclose(rows["pooled_oof_spearman"], pooled, rtol=1e-12, atol=1e-12):
            raise ValueError("grouped secondary metric does not replay.")


def _objective_score(
    values: np.ndarray,
    objective: str,
    parameters: dict[str, object],
    view_id: str,
    protocol: MultistateBehaviorShadowProtocol,
) -> float:
    view = next(item for item in protocol.target_views if item.id == view_id)
    matrix = values.reshape(1, -1)
    if objective == protocol.objective_name:
        return float(
            score_multistate_response_behavior(
                matrix,
                state_ids=protocol.state_ids,
                target_mask=view.target_mask,
                normalization={
                    "response_scale": float(parameters["response_scale"]),
                    "signal_scale": float(parameters["signal_scale"]),
                },
            ).behavior_score[0]
        )
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
    return float(
        score_response_magnitude_feasibility(
            matrix,
            target_mask=view.target_mask,
            calibration=calibration,
        ).feasibility_margin[0]
    )


def _canonical_vector_json(value: object) -> str:
    return json.dumps(_vector(value).tolist(), allow_nan=False, separators=(",", ":"))


def _vector(value: object) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (8,) or not np.isfinite(vector).all():
        raise ValueError("grouped objective validation vectors must contain eight finite values.")
    return vector


def _candidate_set_sha256(candidate_ids: frozenset[str]) -> str:
    canonical = json.dumps(sorted(candidate_ids), ensure_ascii=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _positive_quantile(values: pd.Series, quantile: float) -> float:
    array = values.to_numpy(dtype=float)
    result = float(np.quantile(array, quantile, method="linear")) if array.size else float("nan")
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("grouped fold normalization replay produced an invalid scale.")
    return result


def _optional_spearman(left: pd.Series, right: pd.Series) -> float | None:
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return None
    result = left.corr(right, method="spearman")
    return float(result) if np.isfinite(result) else None


__all__ = ["verify_grouped_objective_validation"]
