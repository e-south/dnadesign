"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_multistate_behavior_completion_gate.py

Completion-gate evidence tests for the multistate behavior shadow protocol.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.core.contracts import (
    StressTargetView,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    multistate_behavior_allocation as behavior_allocation,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    multistate_behavior_grouped_validation as grouped_validation,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    multistate_behavior_normalization_sensitivity as normalization_sensitivity,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    multistate_behavior_protocol as behavior_protocol,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime import (
    multistate_behavior_reference as behavior_reference,
)

PACKAGE = Path("src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy")
PROTOCOL = behavior_protocol.load_multistate_behavior_protocol(
    PACKAGE / "config/multistate_response_behavior_shadow_v1.yaml"
)


def test_completion_gate_is_protocol_declared_and_shadow_only() -> None:
    gate = PROTOCOL.completion_gate

    assert gate.normalization_quantiles == (0.50, 0.75, 0.90, 0.95, 0.99)
    assert gate.normalization_primary_quantile == 0.90
    assert gate.normalization_holdout == "leave_one_source_experiment_out"
    assert gate.validation_label_source == "verified_observed_label_promotion"
    assert gate.validation_seeds == (3, 7, 19, 29, 43)
    assert gate.validation_split == "leave_one_label_source_experiment_out"
    assert gate.validation_x_preprocessing == "identity_train_fold_only"
    assert gate.validation_scoring_parameters == "train_fold_only_exclude_heldout_experiment_and_candidates"
    assert gate.allocation_expected_unique_count == 18
    assert PROTOCOL.campaign_activation == "prohibited"
    assert PROTOCOL.synthesis_authorization == "prohibited"


def test_normalization_sensitivity_covers_quantiles_and_source_holdouts() -> None:
    response, signal, predictions = _normalization_inputs()

    table = normalization_sensitivity.build_multistate_behavior_normalization_sensitivity(
        response_resolution_rows=response,
        signal_resolution_rows=signal,
        predictions=predictions,
        protocol=PROTOCOL,
        target_views=_target_views(),
        normalization_source_rows_sha256="sha256:" + "a" * 64,
        prediction_run_id="run-1",
        prediction_source_sha256="sha256:" + "b" * 64,
    )

    assert set(table["scenario_kind"]) == {"scale_quantile", "leave_one_source_experiment_out"}
    quantiles = table.loc[table["scenario_kind"].eq("scale_quantile"), "scale_quantile"].unique()
    assert tuple(sorted(quantiles)) == PROTOCOL.completion_gate.normalization_quantiles
    holdouts = table.loc[
        table["scenario_kind"].eq("leave_one_source_experiment_out"),
        "excluded_reader_experiment_id",
    ].unique()
    assert set(holdouts) == {"experiment-a", "experiment-b", "experiment-c"}
    assert len(table) == (5 + 3) * 3
    assert table["softmin_scale"].gt(0.0).all()
    assert table["raw_top_k_overlap"].between(0, 6).all()


def test_allocation_comparison_delegates_unique_sequence_allocation() -> None:
    ids = [f"candidate-{index:02d}" for index in range(24)]
    detail_rows: list[dict[str, object]] = []
    for view_index, view_id in enumerate(("ethanol", "ciprofloxacin", "and")):
        shifted = ids[view_index:] + ids[:view_index]
        for rank, candidate_id in enumerate(shifted, start=1):
            detail_rows.append(
                {
                    "id": candidate_id,
                    "selection_view_id": view_id,
                    "hard_score": float(100 - rank),
                    "hard_rank": rank,
                    "behavior_score": float(200 - rank),
                    "behavior_rank": rank,
                    "prediction_run_id": "run-1",
                    "prediction_source_sha256": "sha256:" + "b" * 64,
                    "protocol_id": PROTOCOL.protocol_id,
                    "protocol_source_sha256": "sha256:" + PROTOCOL.source_sha256,
                    "normalization_source_rows_sha256": "sha256:" + "a" * 64,
                }
            )
    candidate_rows = pd.DataFrame(
        {
            "id": ids,
            "sequence": [f"ACGT{index:020d}" for index in range(len(ids))],
        }
    )

    table = behavior_allocation.build_multistate_behavior_allocation_comparison(
        hard_behavior_detail=pd.DataFrame.from_records(detail_rows),
        candidate_records=candidate_rows,
        protocol=PROTOCOL,
    )

    assert len(table) == 36
    assert table.groupby("objective_name")["id"].nunique().eq(18).all()
    assert table.groupby("objective_name")["sequence_sha256"].nunique().eq(18).all()
    assert set(table["allocation_strategy"]) == {"round_robin_next_best_unallocated"}
    assert set(table["evidence_role"]) == {
        "same_fixed_prediction_sequence_deduplicated_allocation_preview_no_campaign_mutation"
    }

    null_candidate_rows = candidate_rows.copy()
    null_candidate_rows.loc[0, "sequence"] = None
    with pytest.raises(ValueError, match="must be non-null"):
        behavior_allocation.build_multistate_behavior_allocation_comparison(
            hard_behavior_detail=pd.DataFrame.from_records(detail_rows),
            candidate_records=null_candidate_rows,
            protocol=PROTOCOL,
        )


def test_grouped_validation_uses_fold_local_parameters_for_both_objectives() -> None:
    labels, x, response, signal, rmf_uncertainty = _grouped_inputs()
    one_seed = replace(
        PROTOCOL,
        completion_gate=replace(PROTOCOL.completion_gate, validation_seeds=(3,)),
    )

    table = grouped_validation.build_grouped_objective_validation(
        labels=labels,
        x=x,
        response_resolution_rows=response,
        signal_resolution_rows=signal,
        rmf_uncertainty_rows=rmf_uncertainty,
        bootstrap_samples=100,
        protocol=one_seed,
        target_views=_target_views(),
        model_params=_model_params(),
        source=_label_source(),
    )

    assert len(table) == len(labels) * 3 * 2
    assert set(table["objective_name"]) == {
        "multistate_response_behavior_v1",
        "response_magnitude_feasibility_v1",
    }
    assert table.groupby(["seed", "selection_view_id", "objective_name"])["pooled_oof_spearman"].nunique().eq(1).all()
    assert table["normalization_parameters_json"].str.contains("excluded_source_experiment").all()
    assert set(table["x_preprocessing"]) == {"identity_train_fold_only"}
    assert set(table["y_fit_space"]) == {"raw_reader_response_window_vector_v1"}


def test_grouped_validation_excludes_repeated_candidates_from_fold_scales() -> None:
    labels, x, response, signal, rmf_uncertainty = _grouped_inputs()
    heldout_candidate = str(labels.loc[0, "candidate_id"])
    repeated_response = response.loc[response["candidate_id"].eq(heldout_candidate)].copy()
    repeated_response["id"] = "repeated-unit"
    repeated_response["reader_experiment_id"] = "experiment-b"
    repeated_response["bootstrap_sd"] = 999.0
    response = pd.concat([response, repeated_response], ignore_index=True)
    repeated_signal = signal.loc[signal["candidate_id"].eq(heldout_candidate)].copy()
    repeated_signal["id"] = "repeated-unit"
    repeated_signal["reader_experiment_id"] = "experiment-b"
    repeated_signal["bootstrap_sd"] = 999.0
    signal = pd.concat([signal, repeated_signal], ignore_index=True)
    repeated_rmf = (
        rmf_uncertainty.loc[rmf_uncertainty["reader_experiment_id"].eq("experiment-a")]
        .drop_duplicates("selection_view_id")
        .copy()
    )
    repeated_rmf["id"] = "repeated-unit"
    repeated_rmf["reader_experiment_id"] = "experiment-b"
    for component in ("response_separation", "on_magnitude_floor", "off_magnitude_ceiling"):
        repeated_rmf[f"{component}__combined_sd"] = 999.0
    rmf_uncertainty = pd.concat([rmf_uncertainty, repeated_rmf], ignore_index=True)
    one_seed = replace(PROTOCOL, completion_gate=replace(PROTOCOL.completion_gate, validation_seeds=(3,)))

    table = grouped_validation.build_grouped_objective_validation(
        labels=labels,
        x=x,
        response_resolution_rows=response,
        signal_resolution_rows=signal,
        rmf_uncertainty_rows=rmf_uncertainty,
        bootstrap_samples=100,
        protocol=one_seed,
        target_views=_target_views(),
        model_params=_model_params(),
        source=_label_source(),
    )

    heldout = table.loc[table["label_source_reader_experiment_id"].eq("experiment-a")]
    parameters = heldout["normalization_parameters_json"].map(json.loads)
    assert parameters.map(lambda value: value["excluded_candidate_count"]).eq(3).all()
    assert parameters.map(lambda value: value["excluded_normalization_unit_count"]).ge(3).all()
    for record in parameters:
        for field, value in record.items():
            if field.endswith("_scale"):
                assert float(value) < 10.0


def test_reference_relative_signal_bootstrap_must_be_definitionally_zero() -> None:
    designs = pd.DataFrame(
        {
            "experiment_id": ["experiment-a"],
            "design_id": ["pDual-10"],
            "reduction_id": ["event_logmean_4_8h_post"],
            "is_reference": [True],
            **{f"b{state}": [0.0] for state in PROTOCOL.state_ids},
            **{f"b{state}_bootstrap_sd": [0.0] for state in PROTOCOL.state_ids},
        }
    )
    draws = pd.DataFrame(
        {
            "experiment_id": ["experiment-a"] * 3,
            "design_id": ["pDual-10"] * 3,
            "reduction_id": ["event_logmean_4_8h_post"] * 3,
            "is_reference": [True] * 3,
            "draw_index": [0, 1, 2],
            **{f"b{state}": [0.0, 0.0, 0.0] for state in PROTOCOL.state_ids},
        }
    )

    receipt = behavior_reference.verify_reference_relative_bootstrap_identity(
        designs,
        draws,
        primary_reduction_id=PROTOCOL.primary_reduction_id,
        state_ids=PROTOCOL.state_ids,
    )
    assert receipt.reference_unit_count == 1
    assert receipt.bootstrap_row_count == 3

    drifted = draws.copy()
    drifted.loc[1, "b10"] = 0.1
    with pytest.raises(ValueError, match="definitionally zero"):
        behavior_reference.verify_reference_relative_bootstrap_identity(
            designs,
            drifted,
            primary_reduction_id=PROTOCOL.primary_reduction_id,
            state_ids=PROTOCOL.state_ids,
        )


def _normalization_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(12)
    units = [(f"unit-{index}", f"experiment-{chr(97 + index % 3)}") for index in range(9)]
    response = pd.DataFrame.from_records(
        {
            "id": unit_id,
            "candidate_id": f"candidate-{index:02d}",
            "reader_experiment_id": experiment,
            "bootstrap_sd": float(0.1 + 0.02 * pair + 0.01 * index),
        }
        for index, (unit_id, experiment) in enumerate(units)
        for pair in range(6)
    )
    signal = pd.DataFrame.from_records(
        {
            "id": unit_id,
            "candidate_id": f"candidate-{index:02d}",
            "reader_experiment_id": experiment,
            "bootstrap_sd": float(0.08 + 0.015 * state + 0.01 * index),
        }
        for index, (unit_id, experiment) in enumerate(units)
        for state in range(4)
    )
    predictions = pd.DataFrame(rng.normal(size=(12, 8)), columns=_components())
    predictions.insert(0, "id", [f"candidate-{index:02d}" for index in range(len(predictions))])
    return response, signal, predictions


def _grouped_inputs() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(37)
    count = 9
    groups = np.asarray(["experiment-a"] * 3 + ["experiment-b"] * 3 + ["experiment-c"] * 3)
    x = rng.normal(size=(count, 5))
    y = np.column_stack([x[:, 0] + 0.05 * index * x[:, 1] for index in range(8)])
    labels = pd.DataFrame(y, columns=_components())
    labels.insert(0, "label_source_reader_experiment_id", groups)
    labels.insert(0, "display_label", [f"Candidate {index}" for index in range(count)])
    labels.insert(0, "candidate_id", [f"candidate-{index:02d}" for index in range(count)])
    response, signal, _ = _normalization_inputs()
    rmf_records: list[dict[str, object]] = []
    for view_id in ("ethanol", "ciprofloxacin", "and"):
        for index, (unit_id, experiment) in enumerate(
            response[["id", "reader_experiment_id"]].drop_duplicates().itertuples(index=False, name=None)
        ):
            row = {"id": unit_id, "selection_view_id": view_id, "reader_experiment_id": experiment}
            for offset, component in enumerate(("response_separation", "on_magnitude_floor", "off_magnitude_ceiling")):
                row[f"{component}__combined_sd"] = 0.2 + 0.01 * index + 0.02 * offset
            rmf_records.append(row)
    return labels, x, response, signal, pd.DataFrame.from_records(rmf_records)


def _components() -> list[str]:
    return ["r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11"]


def _target_views() -> tuple[StressTargetView, ...]:
    return (
        StressTargetView("ethanol", "Ethanol", (0.0, 1.0, 0.0, 1.0)),
        StressTargetView("ciprofloxacin", "Ciprofloxacin", (0.0, 0.0, 1.0, 1.0)),
        StressTargetView("and", "AND", (0.0, 0.0, 0.0, 1.0)),
    )


def _model_params() -> dict[str, object]:
    return {
        "n_estimators": 100,
        "criterion": "friedman_mse",
        "bootstrap": True,
        "oob_score": True,
        "random_state": 7,
        "n_jobs": -1,
        "emit_feature_importance": True,
    }


def _label_source() -> dict[str, str]:
    return {
        "promotion_manifest_sha256": "sha256:" + "c" * 64,
        "candidate_records_sha256": "sha256:" + "d" * 64,
        "source_observation_manifest_sha256": "sha256:" + "e" * 64,
        "x_column_name": "test_x",
    }
