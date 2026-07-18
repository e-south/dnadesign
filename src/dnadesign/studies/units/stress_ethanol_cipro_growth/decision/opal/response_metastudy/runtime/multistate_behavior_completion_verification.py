"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_completion_verification.py

Fail-closed orchestration for behavior completion-gate replay.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dnadesign.opal import score_multistate_response_behavior

from ..core.contracts import StressTargetView
from ..evaluation.multistate_behavior_cardinality import build_family_cardinality_pressure
from ..evaluation.multistate_behavior_comparison import compare_hard_and_behavior_scores
from ..evaluation.multistate_behavior_protocol import MultistateBehaviorShadowProtocol
from ..evaluation.multistate_behavior_rmf_replay import build_current_rmf_prediction_scores
from .multistate_behavior_allocation_verification import verify_allocation_comparison
from .multistate_behavior_frame_verification import assert_frame_equal_by_key
from .multistate_behavior_grouped_verification import verify_grouped_objective_validation
from .multistate_behavior_semantic_verification import BehaviorBundleSemantics
from .multistate_behavior_sensitivity_verification import verify_normalization_sensitivity


def verify_behavior_completion_tables(
    tables: dict[str, pd.DataFrame],
    *,
    semantics: BehaviorBundleSemantics,
    protocol: MultistateBehaviorShadowProtocol,
    reader_bundle_manifest_sha256: str,
) -> None:
    """Replay normalization, grouped scoring, current RMF, and allocation evidence."""

    _verify_completion_provenance(tables, semantics=semantics, protocol=protocol)
    _verify_prediction_scores(tables, semantics=semantics, protocol=protocol)
    verify_normalization_sensitivity(tables, semantics=semantics, protocol=protocol)
    verify_grouped_objective_validation(tables, protocol=protocol)
    _verify_observed_control_face_validity(tables, protocol=protocol)
    _assert_frame(
        tables["family_cardinality_pressure"],
        build_family_cardinality_pressure(protocol),
        ["state_count"],
    )
    _verify_rmf_replay(
        tables,
        semantics=semantics,
        protocol=protocol,
        reader_bundle_manifest_sha256=reader_bundle_manifest_sha256,
    )
    verify_allocation_comparison(tables, protocol=protocol)


def _verify_completion_provenance(
    tables: dict[str, pd.DataFrame],
    *,
    semantics: BehaviorBundleSemantics,
    protocol: MultistateBehaviorShadowProtocol,
) -> None:
    expected = {
        "protocol_id": protocol.protocol_id,
        "protocol_source_sha256": semantics.protocol_sha256,
    }
    for table_id in ("normalization_sensitivity", "grouped_objective_validation", "allocation_comparison"):
        frame = tables[table_id]
        for field, value in expected.items():
            if set(frame[field].astype(str)) != {str(value)}:
                raise ValueError(f"completion table {table_id!r} {field} provenance drifted.")
    for table_id in ("normalization_sensitivity", "allocation_comparison"):
        frame = tables[table_id]
        if set(frame["normalization_source_rows_sha256"].astype(str)) != {semantics.source_rows_sha256}:
            raise ValueError(f"completion table {table_id!r} normalization provenance drifted.")
        if set(frame["prediction_run_id"].astype(str)) != {semantics.prediction_run_id}:
            raise ValueError(f"completion table {table_id!r} prediction run drifted.")
        if set(frame["prediction_source_sha256"].astype(str)) != {semantics.prediction_source_sha256}:
            raise ValueError(f"completion table {table_id!r} prediction source drifted.")
    vectors = tables["prediction_vectors"]
    if len(vectors) != semantics.prediction_count or vectors["id"].astype(str).duplicated().any():
        raise ValueError("fixed prediction-vector coverage drifted.")
    if set(vectors["prediction_run_id"].astype(str)) != {semantics.prediction_run_id}:
        raise ValueError("fixed prediction-vector run provenance drifted.")
    if set(vectors["prediction_source_sha256"].astype(str)) != {semantics.prediction_source_sha256}:
        raise ValueError("fixed prediction-vector source provenance drifted.")
    if vectors["sequence_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").ne(True).any():
        raise ValueError("fixed prediction-vector sequence digests are invalid.")
    grouped = tables["grouped_objective_validation"]
    expected_model_digest = "sha256:" + protocol.completion_gate.validation_model_nonseed_params_sha256
    if set(grouped["configured_model_params_sha256"].astype(str)) != {expected_model_digest}:
        raise ValueError("grouped validation registered model contract drifted.")


def _verify_prediction_scores(
    tables: dict[str, pd.DataFrame],
    *,
    semantics: BehaviorBundleSemantics,
    protocol: MultistateBehaviorShadowProtocol,
) -> None:
    vectors = tables["prediction_vectors"]
    scores = tables["prediction_scores"]
    matrix = vectors.loc[:, _components(protocol)].to_numpy(dtype=float)
    for view in protocol.target_views:
        result = score_multistate_response_behavior(
            matrix,
            state_ids=protocol.state_ids,
            target_mask=view.target_mask,
            normalization={"response_scale": semantics.response_scale, "signal_scale": semantics.signal_scale},
        )
        observed = scores.loc[scores["selection_view_id"].astype(str).eq(view.id)].set_index("id").loc[vectors["id"]]
        expected = {
            "behavior_score": result.behavior_score,
            "hard_bottleneck_clearance": result.hard_bottleneck_clearance,
            "response_family_score": result.response_family_score,
            "on_signal_family_score": result.on_signal_family_score,
            "off_signal_suppression_family_score": result.off_signal_suppression_family_score,
        }
        for field, values in expected.items():
            if not np.allclose(observed[field].to_numpy(dtype=float), values, rtol=1e-12, atol=1e-12):
                raise ValueError(f"prediction {field} does not replay from fixed raw vectors.")
        if tuple(observed["limiting_coordinate"].astype(str)) != tuple(result.limiting_coordinate_label):
            raise ValueError("prediction limiting coordinates do not replay from fixed raw vectors.")
        if not np.array_equal(
            observed["all_reference_directions_met"].to_numpy(dtype=bool),
            result.all_reference_directions_met,
        ):
            raise ValueError("prediction natural-zero diagnostic does not replay from fixed raw vectors.")


def _verify_observed_control_face_validity(
    tables: dict[str, pd.DataFrame],
    *,
    protocol: MultistateBehaviorShadowProtocol,
) -> None:
    controls = tables["observed_control_face_validity"]
    observed = tables["observed_scores"]
    expected_controls = {
        (control.selection_view_id, control.design_id, control.display_label)
        for control in protocol.completion_gate.face_validity_controls
    }
    if set(controls[["selection_view_id", "design_id", "display_label"]].itertuples(index=False, name=None)) != (
        expected_controls
    ):
        raise ValueError("observed biological face-validity controls drifted from the study protocol.")
    if set(controls["selection_view_id"].astype(str)) & set(
        protocol.completion_gate.face_validity_unclaimed_positive_control_views
    ):
        raise ValueError("face-validity evidence claims a positive control for a deliberately unclaimed view.")
    score_fields = (
        "behavior_score",
        "hard_bottleneck_clearance",
        "response_family_score",
        "on_signal_family_score",
        "off_signal_suppression_family_score",
    )
    for row in controls.itertuples(index=False):
        match = observed.loc[
            observed["id"].astype(str).eq(str(row.id))
            & observed["selection_view_id"].astype(str).eq(str(row.selection_view_id))
        ]
        if len(match) != 1:
            raise ValueError("face-validity row does not identify one observed unit/view score.")
        source = match.iloc[0]
        if str(source["candidate_id"]) != str(row.candidate_id) or str(source["reader_experiment_id"]) != str(
            row.reader_experiment_id
        ):
            raise ValueError("face-validity candidate or experiment identity disagrees with observed evidence.")
        if any(
            not np.isclose(float(getattr(row, field)), float(source[field]), rtol=1e-12, atol=1e-12)
            for field in score_fields
        ):
            raise ValueError("face-validity score fields do not replay from observed evidence.")
        if str(row.limiting_coordinate) != str(source["limiting_coordinate"]) or bool(
            row.all_reference_directions_met
        ) != bool(source["all_reference_directions_met"]):
            raise ValueError("face-validity diagnostic fields do not replay from observed evidence.")
        view = observed.loc[observed["selection_view_id"].astype(str).eq(str(row.selection_view_id))]
        ranking = view.sort_values(["behavior_score", "id"], ascending=[False, True], kind="mergesort")["id"].astype(
            str
        )
        expected_rank = ranking.tolist().index(str(row.id)) + 1
        if int(row.observed_unit_rank) != expected_rank or int(row.observed_unit_count) != len(view):
            raise ValueError("face-validity rank or support count does not replay.")
        if str(row.evidence_role) != protocol.completion_gate.face_validity_evidence_role:
            raise ValueError("face-validity evidence role drifted.")


def _verify_rmf_replay(
    tables: dict[str, pd.DataFrame],
    *,
    semantics: BehaviorBundleSemantics,
    protocol: MultistateBehaviorShadowProtocol,
    reader_bundle_manifest_sha256: str,
) -> None:
    calibration = tables["rmf_replay_calibration"]
    resolution = tables["grouped_rmf_resolution"]
    _verify_rmf_resolution_coverage(tables, protocol=protocol)
    if len(calibration) != 3 * len(protocol.target_views) or not calibration["threshold"].eq(0.0).all():
        raise ValueError("RMF replay calibration must contain three zero-threshold requirements per view.")
    expected_literals = {
        "scale_quantile": protocol.completion_gate.normalization_primary_quantile,
        "bootstrap_samples": int(tables["normalization_response_resolution"]["bootstrap_samples"].iloc[0]),
        "excluded_experiment": None,
        "scale_basis": "reader_joint_bootstrap_plus_conservative_event_bound",
        "reader_bundle_manifest_sha256": reader_bundle_manifest_sha256,
        "normalization_source_rows_sha256": semantics.source_rows_sha256,
        "evidence_role": "corrected_reader_calibration_replay_same_fixed_raw_prediction_matrix",
    }
    for field, expected in expected_literals.items():
        values = calibration[field]
        if expected is None:
            if not values.isna().all():
                raise ValueError(f"RMF replay calibration {field!r} drifted.")
        elif set(values.astype(str)) != {str(expected)}:
            raise ValueError(f"RMF replay calibration {field!r} drifted.")
    quantile = protocol.completion_gate.normalization_primary_quantile
    for row in calibration.itertuples(index=False):
        source = resolution.loc[
            resolution["selection_view_id"].astype(str).eq(str(row.selection_view_id)),
            f"{row.component}__combined_sd",
        ].to_numpy(dtype=float)
        expected_scale = float(np.quantile(source, quantile, method="linear"))
        if not np.isclose(float(row.scale), expected_scale, rtol=1e-12, atol=1e-12):
            raise ValueError("RMF replay calibration scale does not derive from corrected Reader evidence.")
    views = tuple(StressTargetView(view.id, view.id, view.target_mask) for view in protocol.target_views)
    hard = build_current_rmf_prediction_scores(
        predictions=tables["prediction_vectors"],
        calibration=calibration,
        protocol=protocol,
        target_views=views,
    )
    comparison = compare_hard_and_behavior_scores(
        hard,
        tables["prediction_scores"],
        top_k=protocol.prediction_raw_top_k,
        hard_score_semantics=(
            f"{protocol.comparator_objective_name}.{protocol.comparator_score_channel}.{protocol.comparator_direction}"
        ),
    )
    _assert_frame(tables["hard_behavior_summary"], comparison.summary, ["selection_view_id"])
    _assert_frame(tables["hard_behavior_detail"], comparison.detail, ["selection_view_id", "id"])


def _verify_rmf_resolution_coverage(
    tables: dict[str, pd.DataFrame],
    *,
    protocol: MultistateBehaviorShadowProtocol,
) -> None:
    resolution = tables["grouped_rmf_resolution"]
    response = tables["normalization_response_resolution"]
    units = response.loc[:, ["id", "reader_experiment_id"]].drop_duplicates().set_index("id")
    if resolution.duplicated(subset=["id", "selection_view_id"]).any():
        raise ValueError("RMF resolution evidence contains duplicate unit/view rows.")
    expected_product = pd.MultiIndex.from_product(
        [units.index.astype(str), [view.id for view in protocol.target_views]],
        names=["id", "selection_view_id"],
    )
    observed_product = pd.MultiIndex.from_frame(resolution.loc[:, ["id", "selection_view_id"]].astype(str))
    if set(observed_product) != set(expected_product):
        raise ValueError("RMF resolution evidence does not cover the exact normalization unit/view product.")
    expected_experiments = units["reader_experiment_id"].astype(str)
    observed_experiments = resolution.set_index("id")["reader_experiment_id"].astype(str)
    if not observed_experiments.groupby(level=0).nunique().eq(1).all():
        raise ValueError("RMF resolution experiment identity drifts across views.")
    if observed_experiments.groupby(level=0).first().to_dict() != expected_experiments.to_dict():
        raise ValueError("RMF resolution experiment identity disagrees with normalization evidence.")
    columns = [
        f"{component}__combined_sd"
        for component in ("response_separation", "on_magnitude_floor", "off_magnitude_ceiling")
    ]
    values = resolution.loc[:, columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("RMF resolution combined uncertainties must be finite and positive.")


def _components(protocol: MultistateBehaviorShadowProtocol) -> list[str]:
    return [f"{prefix}{state}" for prefix in ("r", "b") for state in protocol.state_ids]


def _assert_frame(observed: pd.DataFrame, expected: pd.DataFrame, keys: list[str]) -> None:
    try:
        assert_frame_equal_by_key(observed, expected, keys=keys)
    except AssertionError as exc:
        raise ValueError("completion-gate table does not replay from persisted lower-level evidence.") from exc


__all__ = ["verify_behavior_completion_tables"]
