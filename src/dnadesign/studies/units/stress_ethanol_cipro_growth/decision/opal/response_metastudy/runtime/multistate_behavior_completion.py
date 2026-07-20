"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_completion.py

Study-owned completion evidence for the multistate behavior shadow.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..core.contracts import StressTargetView
from ..evaluation.multistate_behavior_allocation import build_multistate_behavior_allocation_comparison
from ..evaluation.multistate_behavior_cardinality import build_family_cardinality_pressure
from ..evaluation.multistate_behavior_face_validity import build_behavior_face_validity
from ..evaluation.multistate_behavior_grouped_validation import build_grouped_objective_validation
from ..evaluation.multistate_behavior_normalization import MultistateBehaviorNormalizationEvidence
from ..evaluation.multistate_behavior_normalization_sensitivity import (
    build_multistate_behavior_normalization_sensitivity,
)
from ..evaluation.multistate_behavior_protocol import MultistateBehaviorShadowProtocol
from .multistate_behavior_labels import (
    VerifiedBehaviorValidationLabels,
    load_verified_behavior_validation_labels,
)


@dataclass(frozen=True)
class MultistateBehaviorCompletionEvidence:
    """Three prespecified analyses needed for a study decision."""

    normalization_sensitivity: pd.DataFrame
    grouped_objective_validation: pd.DataFrame
    allocation_comparison: pd.DataFrame
    observed_control_face_validity: pd.DataFrame
    family_cardinality_pressure: pd.DataFrame
    validation_labels: VerifiedBehaviorValidationLabels
    rmf_resolution_rows: pd.DataFrame
    rmf_replay_calibration: pd.DataFrame
    prediction_vectors: pd.DataFrame


def build_multistate_behavior_completion_evidence(
    *,
    normalization: MultistateBehaviorNormalizationEvidence,
    predictions: pd.DataFrame,
    observed_scores: pd.DataFrame,
    hard_behavior_detail: pd.DataFrame,
    candidate_records_path: Path,
    campaign_config_path: Path,
    current_measurements: pd.DataFrame,
    source_observation_bundle_root: Path,
    rmf_uncertainty_rows: pd.DataFrame,
    rmf_replay_calibration: pd.DataFrame,
    target_views: tuple[StressTargetView, ...],
    protocol: MultistateBehaviorShadowProtocol,
    model_params: dict[str, object],
    prediction_run_id: str,
    prediction_source_sha256: str,
) -> MultistateBehaviorCompletionEvidence:
    """Build normalization, prediction-to-truth, and allocation evidence."""

    sensitivity = build_multistate_behavior_normalization_sensitivity(
        response_resolution_rows=normalization.response_resolution_rows,
        signal_resolution_rows=normalization.signal_resolution_rows,
        predictions=predictions,
        protocol=protocol,
        target_views=target_views,
        normalization_source_rows_sha256=f"sha256:{normalization.source_rows_sha256}",
        prediction_run_id=prediction_run_id,
        prediction_source_sha256=prediction_source_sha256,
    )
    validation_labels = load_verified_behavior_validation_labels(
        campaign_config_path=campaign_config_path,
        current_measurements=current_measurements,
        source_observation_bundle_root=source_observation_bundle_root,
        protocol=protocol,
    )
    grouped = build_grouped_objective_validation(
        labels=validation_labels.labels,
        x=validation_labels.x,
        response_resolution_rows=normalization.response_resolution_rows,
        signal_resolution_rows=normalization.signal_resolution_rows,
        rmf_uncertainty_rows=rmf_uncertainty_rows,
        bootstrap_samples=normalization.bootstrap_samples,
        protocol=protocol,
        target_views=target_views,
        model_params=model_params,
        source=validation_labels.source,
    )
    candidate_records = pd.read_parquet(
        candidate_records_path,
        columns=["id", "sequence", "usr_label__primary"],
    )
    allocation = build_multistate_behavior_allocation_comparison(
        hard_behavior_detail=hard_behavior_detail,
        candidate_records=candidate_records,
        protocol=protocol,
    )
    face_validity = build_behavior_face_validity(
        observed_scores,
        current_measurements,
        protocol=protocol,
    )
    prediction_vectors = _prediction_vector_projection(
        predictions,
        candidate_records=candidate_records,
        protocol=protocol,
    )
    return MultistateBehaviorCompletionEvidence(
        normalization_sensitivity=sensitivity,
        grouped_objective_validation=grouped,
        allocation_comparison=allocation,
        observed_control_face_validity=face_validity,
        family_cardinality_pressure=build_family_cardinality_pressure(
            protocol,
            softmin_scale=normalization.softmin_scale,
        ),
        validation_labels=validation_labels,
        rmf_resolution_rows=_rmf_resolution_projection(rmf_uncertainty_rows),
        rmf_replay_calibration=rmf_replay_calibration,
        prediction_vectors=prediction_vectors,
    )


def _prediction_vector_projection(
    frame: pd.DataFrame,
    *,
    candidate_records: pd.DataFrame,
    protocol: MultistateBehaviorShadowProtocol,
) -> pd.DataFrame:
    components = tuple(f"{prefix}{state}" for prefix in ("r", "b") for state in protocol.state_ids)
    required = {"id", "prediction_run_id", "prediction_source_sha256", *components}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"fixed prediction vector projection lacks fields: {missing}")
    rows = frame.loc[:, ["id", "prediction_run_id", "prediction_source_sha256", *components]].copy()
    if rows.empty or rows["id"].astype(str).duplicated().any():
        raise ValueError("fixed prediction vector projection requires unique candidate IDs.")
    candidate_rows = candidate_records.loc[:, ["id", "sequence", "usr_label__primary"]].copy()
    if candidate_rows[["id", "sequence"]].isna().any().any():
        raise ValueError("fixed prediction candidate IDs and sequences must be non-null.")
    candidate_rows["id"] = candidate_rows["id"].astype(str)
    candidate_rows["sequence"] = candidate_rows["sequence"].astype(str)
    if (
        candidate_rows["id"].str.strip().ne(candidate_rows["id"]).any()
        or candidate_rows["sequence"].str.strip().ne(candidate_rows["sequence"]).any()
        or candidate_rows["id"].eq("").any()
        or candidate_rows["sequence"].eq("").any()
    ):
        raise ValueError("fixed prediction candidate IDs and sequences must be exact and nonempty.")
    candidate_rows["sequence_sha256"] = (
        candidate_rows["sequence"].astype(str).map(lambda value: hashlib.sha256(value.encode("ascii")).hexdigest())
    )
    candidate_rows["display_label"] = candidate_rows["usr_label__primary"].astype("string").fillna("")
    candidate_rows["display_label"] = candidate_rows["display_label"].where(
        candidate_rows["display_label"].str.strip().ne(""),
        candidate_rows["id"].astype(str).str.slice(0, 10),
    )
    rows = rows.merge(
        candidate_rows.loc[:, ["id", "display_label", "sequence_sha256"]],
        on="id",
        how="left",
        validate="one_to_one",
    )
    if rows[["display_label", "sequence_sha256"]].isna().any().any():
        raise ValueError("fixed prediction vector projection lacks candidate identity metadata.")
    rows = rows.loc[
        :,
        [
            "id",
            "display_label",
            "sequence_sha256",
            "prediction_run_id",
            "prediction_source_sha256",
            *components,
        ],
    ]
    rows["evidence_role"] = "fixed_raw_response_window_prediction_for_objective_replay"
    return rows


def _rmf_resolution_projection(frame: pd.DataFrame) -> pd.DataFrame:
    components = ("response_separation", "on_magnitude_floor", "off_magnitude_ceiling")
    columns = [
        "id",
        "selection_view_id",
        "reader_experiment_id",
        *(f"{component}__combined_sd" for component in components),
    ]
    if missing := sorted(set(columns) - set(frame.columns)):
        raise ValueError(f"RMF grouped-validation resolution evidence lacks fields: {missing}")
    rows = frame.loc[:, columns].copy()
    if rows.empty or rows.duplicated(subset=["id", "selection_view_id"]).any():
        raise ValueError("RMF grouped-validation resolution evidence must be unique per unit and view.")
    return rows.sort_values(["selection_view_id", "id"], kind="mergesort").reset_index(drop=True)


__all__ = [
    "MultistateBehaviorCompletionEvidence",
    "build_multistate_behavior_completion_evidence",
]
