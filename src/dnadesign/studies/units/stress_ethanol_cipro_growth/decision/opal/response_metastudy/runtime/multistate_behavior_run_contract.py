"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_run_contract.py

Expected OPAL model, label, objective, and selection lineage for shadow comparison.

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

RESPONSE_Y_COLUMNS = ("r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11")


def verify_behavior_run_receipt(
    receipt: pd.Series,
    *,
    rows: pd.DataFrame,
    prediction_run_id: str,
    state_ids: tuple[str, ...],
    target_masks: Mapping[str, tuple[float, ...]],
    comparator_calibration_by_view: Mapping[str, Mapping[str, float]],
    comparator_objective_name: str,
    comparator_channel: str,
    comparator_direction: str,
    model_name: str,
    model_params: Mapping[str, object],
    raw_top_k: int,
) -> dict[str, object]:
    """Verify a run receipt against the exact loaded campaign contract."""

    if str(receipt.get("event")) != "run_meta" or str(receipt.get("run_id")) != prediction_run_id:
        raise ValueError("prediction run receipt identity is invalid.")
    scored_count = receipt.get("stats__n_scored")
    if isinstance(scored_count, (bool, np.bool_)) or not isinstance(scored_count, (int, np.integer)):
        raise ValueError("prediction run receipt lacks an integer scored count.")
    if int(scored_count) != len(rows):
        raise ValueError(
            f"prediction rows disagree with the run receipt: receipt={int(scored_count)}, rows={len(rows)}."
        )
    if not rows["event"].astype(str).eq("run_pred").all():
        raise ValueError("prediction run contains a non-prediction ledger event.")
    if not rows["as_of_round"].eq(receipt["as_of_round"]).all():
        raise ValueError("prediction rows disagree with the run receipt round.")
    run_lineage = _verify_model_and_label_lineage(
        receipt,
        model_name=model_name,
        model_params=model_params,
    )
    _verify_comparator_definitions(
        _json_list(receipt.get("objective__defs_json"), field="objective__defs_json"),
        selection_defs=_json_list(
            receipt.get("selection_views__defs_json"),
            field="selection_views__defs_json",
        ),
        state_ids=state_ids,
        target_masks=target_masks,
        calibration_by_view=comparator_calibration_by_view,
        objective_name=comparator_objective_name,
        score_channel=comparator_channel,
        direction=comparator_direction,
        candidate_count=len(rows),
        raw_top_k=raw_top_k,
    )
    return run_lineage


def _verify_comparator_definitions(
    objective_defs: list[object],
    *,
    selection_defs: list[object],
    state_ids: tuple[str, ...],
    target_masks: Mapping[str, tuple[float, ...]],
    calibration_by_view: Mapping[str, Mapping[str, float]],
    objective_name: str,
    score_channel: str,
    direction: str,
    candidate_count: int,
    raw_top_k: int,
) -> None:
    objectives = _records_by_view(objective_defs, context="objective definitions")
    selections = _records_by_view(selection_defs, context="selection-view definitions")
    expected_views = set(target_masks)
    if (
        set(objectives) != expected_views
        or set(selections) != expected_views
        or set(calibration_by_view) != expected_views
    ):
        raise ValueError("prediction run selection views disagree with the shadow comparator.")
    for view_id, mask in target_masks.items():
        objective = objectives[view_id]
        selection = selections[view_id]
        expected_ref = f"{view_id}/{score_channel}"
        expected_params = {
            "state_ids": list(state_ids),
            "target_mask": list(mask),
            "calibration": dict(calibration_by_view[view_id]),
        }
        if objective.get("objective_name") != objective_name:
            raise ValueError(f"prediction objective disagrees for selection view {view_id!r}.")
        if _plain_value(objective.get("params")) != _plain_value(expected_params):
            raise ValueError(f"prediction objective parameters disagree for selection view {view_id!r}.")
        expected_channels = {
            expected_ref,
            f"{view_id}/response_separation",
            f"{view_id}/on_magnitude_floor",
            f"{view_id}/off_magnitude_ceiling",
        }
        if set(objective.get("score_channels", ())) != expected_channels:
            raise ValueError(f"prediction run lacks comparator score channel {expected_ref!r}.")
        _verify_selection_definition(
            selection,
            view_id=view_id,
            expected_params=expected_params,
            objective_name=objective_name,
            score_channel=score_channel,
            direction=direction,
            raw_top_k=raw_top_k,
            candidate_count=candidate_count,
        )


def _verify_selection_definition(
    selection: dict[str, object],
    *,
    view_id: str,
    expected_params: dict[str, object],
    objective_name: str,
    score_channel: str,
    direction: str,
    raw_top_k: int,
    candidate_count: int,
) -> None:
    expected_ref = f"{view_id}/{score_channel}"
    if (
        selection.get("objective_name") != objective_name
        or _plain_value(selection.get("objective_params")) != _plain_value(expected_params)
        or selection.get("selection_name") != "top_n"
        or selection.get("score_ref") != expected_ref
        or selection.get("objective_mode") != direction
        or selection.get("tie_handling") != "ordinal"
        or selection.get("top_k") != raw_top_k
    ):
        raise ValueError(f"prediction comparator contract disagrees for {view_id!r}.")
    if selection.get("selection_params") != {
        "top_k": raw_top_k,
        "score_ref": score_channel,
        "tie_handling": "ordinal",
        "objective_mode": direction,
        "exclude_already_labeled": True,
        "require_exact_top_k": True,
    }:
        raise ValueError(f"prediction selection parameters disagree for {view_id!r}.")
    summary = selection.get("objective_summary_stats")
    if not isinstance(summary, dict) or int(summary.get("candidate_count", -1)) != candidate_count:
        raise ValueError(f"prediction scored-count summary disagrees for selection view {view_id!r}.")


def _verify_model_and_label_lineage(
    receipt: pd.Series,
    *,
    model_name: str,
    model_params: Mapping[str, object],
) -> dict[str, object]:
    if receipt.get("model__name") != model_name:
        raise ValueError("prediction run model identity disagrees with the configured campaign.")
    configured = dict(model_params)
    emit_feature_importance = configured.pop("emit_feature_importance", False)
    expected_model_params = RandomForestRegressor(**configured).get_params(deep=False)
    expected_model_params["emit_feature_importance"] = emit_feature_importance
    observed_model_params = receipt.get("model__params")
    if _plain_value(observed_model_params) != _plain_value(expected_model_params):
        raise ValueError("prediction run model parameters disagree with the configured campaign.")
    expected_y_ingest = {
        "id_column": "id",
        "sequence_column": "sequence",
        "value_columns": list(RESPONSE_Y_COLUMNS),
    }
    if (
        receipt.get("x_transform__name") != "identity"
        or _plain_value(receipt.get("x_transform__params")) not in (None, {})
        or receipt.get("y_ingest__name") != "vector_from_table_v1"
        or _plain_value(receipt.get("y_ingest__params")) != expected_y_ingest
        or _plain_value(receipt.get("training__y_ops")) is not None
    ):
        raise ValueError("prediction run X/Y transformation lineage disagrees with the campaign.")
    training_count = receipt.get("stats__n_train")
    if isinstance(training_count, (bool, np.bool_)) or not isinstance(training_count, (int, np.integer)):
        raise ValueError("prediction run receipt lacks an integer training count.")
    if int(training_count) <= 0:
        raise ValueError("prediction run must contain at least one training label.")
    as_of_round = receipt.get("as_of_round")
    if isinstance(as_of_round, (bool, np.bool_)) or not isinstance(as_of_round, (int, np.integer)):
        raise ValueError("prediction run receipt lacks an integer round identity.")
    return {
        "as_of_round": int(as_of_round),
        "model_name": model_name,
        "model_params_sha256": _canonical_value_sha256(observed_model_params),
        "y_ingest_name": "vector_from_table_v1",
        "y_ingest_params_sha256": _canonical_value_sha256(receipt.get("y_ingest__params")),
        "training_y_ops_sha256": _canonical_value_sha256(receipt.get("training__y_ops")),
        "training_row_count": int(training_count),
    }


def _json_list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, str):
        raise ValueError(f"prediction run receipt field {field!r} must be JSON text.")
    parsed = json.loads(value, parse_constant=_reject_json_constant)
    if not isinstance(parsed, list):
        raise ValueError(f"prediction run receipt field {field!r} must decode to a list.")
    return parsed


def _records_by_view(records: list[object], *, context: str) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for value in records:
        if not isinstance(value, dict):
            raise ValueError(f"prediction run {context} must contain mappings.")
        view_id = str(value.get("selection_view_id", ""))
        if not view_id or view_id in result:
            raise ValueError(f"prediction run {context} contain missing or duplicate view ids.")
        result[view_id] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"prediction run receipt contains non-finite JSON value {value!r}.")


def _plain_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_value_sha256(value: object) -> str:
    rendered = json.dumps(_plain_value(value), allow_nan=False, separators=(",", ":"), sort_keys=True)
    return "sha256:" + hashlib.sha256(rendered.encode("utf-8")).hexdigest()


__all__ = ["RESPONSE_Y_COLUMNS", "verify_behavior_run_receipt"]
