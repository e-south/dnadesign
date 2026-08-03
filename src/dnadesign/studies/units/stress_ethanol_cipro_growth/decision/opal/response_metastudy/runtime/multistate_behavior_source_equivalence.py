"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_source_equivalence.py

Typed receipt for corrected Reader evidence and immutable point-label reuse.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .multistate_behavior_shadow import VerifiedMultistateBehaviorShadow

SCHEMA_ID = "stress_ethanol_cipro_growth.multistate_response_behavior_source_equivalence.v1"


def build_source_equivalence_receipt(preview: VerifiedMultistateBehaviorShadow) -> dict[str, object]:
    """Persist the proof that corrected uncertainty does not change promoted point labels."""

    labels = preview.completion.validation_labels
    protocol = preview.normalization.protocol
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": "1",
        "study_id": protocol.study_id,
        "protocol_id": protocol.protocol_id,
        "corrected_reader_bundle_manifest_sha256": preview.source["reader_bundle_manifest_sha256"],
        "prior_observation_reader_bundle_manifest_sha256": "sha256:"
        + protocol.source_equivalence.prior_observation_reader_bundle_sha256,
        "central_label_equivalence_sha256": labels.central_label_equivalence_sha256,
        "label_artifact_sha256": labels.label_artifact_sha256,
        "promotion_manifest_sha256": labels.source["promotion_manifest_sha256"],
        "source_observation_manifest_sha256": labels.source["source_observation_manifest_sha256"],
        "candidate_records_sha256": labels.source["candidate_records_sha256"],
        "promoted_candidate_count": labels.promoted_candidate_count,
        "reference_unit_count": preview.reference_identity.reference_unit_count,
        "reference_descriptive_resampling_row_count": (preview.reference_identity.descriptive_resampling_row_count),
        "normalization_reference_unit_count": 0,
        "central_vectors_exactly_equal": True,
        "new_observation_version_required_for_point_labels": False,
        "claim": "corrected_reference_bootstrap_changes_uncertainty_evidence_not_promoted_central_vectors",
    }


def verify_source_equivalence_receipt(
    receipt: Mapping[str, object],
    *,
    decision_source: Mapping[str, object],
    study_id: str,
    protocol_id: str,
    corrected_reader_bundle_manifest_sha256: str,
    promoted_candidate_count: int,
    grouped_validation: pd.DataFrame,
) -> None:
    """Require exact receipt structure and agreement with the decision projection."""

    expected_fields = {
        "schema_id",
        "schema_version",
        "study_id",
        "protocol_id",
        "corrected_reader_bundle_manifest_sha256",
        "prior_observation_reader_bundle_manifest_sha256",
        "central_label_equivalence_sha256",
        "label_artifact_sha256",
        "promotion_manifest_sha256",
        "source_observation_manifest_sha256",
        "candidate_records_sha256",
        "promoted_candidate_count",
        "reference_unit_count",
        "reference_descriptive_resampling_row_count",
        "normalization_reference_unit_count",
        "central_vectors_exactly_equal",
        "new_observation_version_required_for_point_labels",
        "claim",
    }
    if set(receipt) != expected_fields:
        raise ValueError("source-equivalence receipt fields are incomplete or unexpected.")
    literals = {
        "schema_id": SCHEMA_ID,
        "schema_version": "1",
        "central_vectors_exactly_equal": True,
        "new_observation_version_required_for_point_labels": False,
        "normalization_reference_unit_count": 0,
        "claim": "corrected_reference_bootstrap_changes_uncertainty_evidence_not_promoted_central_vectors",
    }
    if any(receipt.get(field) != value for field, value in literals.items()):
        raise ValueError("source-equivalence receipt semantics drifted.")
    expected_context = {
        "study_id": study_id,
        "protocol_id": protocol_id,
        "corrected_reader_bundle_manifest_sha256": corrected_reader_bundle_manifest_sha256,
        "promoted_candidate_count": promoted_candidate_count,
    }
    if any(receipt.get(field) != value for field, value in expected_context.items()):
        raise ValueError("source-equivalence receipt context disagrees with the verified bundle.")
    expected_equivalence = grouped_central_equivalence_sha256(grouped_validation)
    if receipt.get("central_label_equivalence_sha256") != expected_equivalence:
        raise ValueError("source-equivalence central-label digest does not derive from grouped exact labels.")
    grouped_sources = {
        "promotion_manifest_sha256": "promotion_manifest_sha256",
        "source_observation_manifest_sha256": "source_observation_manifest_sha256",
        "candidate_records_sha256": "candidate_records_sha256",
    }
    for receipt_field, grouped_field in grouped_sources.items():
        values = set(grouped_validation[grouped_field].astype(str))
        if values != {str(receipt[receipt_field])}:
            raise ValueError(f"source-equivalence {receipt_field!r} disagrees with grouped source provenance.")
    for field in (
        "corrected_reader_bundle_manifest_sha256",
        "prior_observation_reader_bundle_manifest_sha256",
        "central_label_equivalence_sha256",
        "label_artifact_sha256",
        "promotion_manifest_sha256",
        "source_observation_manifest_sha256",
        "candidate_records_sha256",
    ):
        value = receipt.get(field)
        if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
            raise ValueError(f"source-equivalence field {field!r} is not a canonical digest.")
    for field in (
        "promoted_candidate_count",
        "reference_unit_count",
        "reference_descriptive_resampling_row_count",
    ):
        value = receipt.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"source-equivalence field {field!r} must be positive integer evidence.")
    shared = {
        "corrected_reader_bundle_manifest_sha256",
        "prior_observation_reader_bundle_manifest_sha256",
        "central_label_equivalence_sha256",
        "label_artifact_sha256",
        "promotion_manifest_sha256",
        "reference_unit_count",
        "reference_descriptive_resampling_row_count",
        "normalization_reference_unit_count",
    }
    if any(decision_source.get(field) != receipt.get(field) for field in shared):
        raise ValueError("decision source equivalence disagrees with its typed receipt.")


def grouped_central_equivalence_sha256(frame: pd.DataFrame) -> str:
    """Recompute the exact candidate/source/vector digest from grouped evidence."""

    required = {"candidate_id", "label_source_reader_experiment_id", "observed_y"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"grouped central-equivalence evidence lacks fields: {missing}")
    rows = frame.loc[:, sorted(required)].copy()
    rows["vector_hex"] = rows["observed_y"].map(_vector_hex)
    rows = rows.drop_duplicates(["candidate_id", "label_source_reader_experiment_id", "vector_hex"])
    if rows["candidate_id"].astype(str).duplicated().any():
        raise ValueError("grouped central-equivalence evidence contains conflicting candidate rows.")
    records = [
        (
            str(row.candidate_id),
            str(row.label_source_reader_experiment_id),
            tuple(row.vector_hex),
        )
        for row in rows.sort_values("candidate_id", kind="mergesort").itertuples(index=False)
    ]
    return "sha256:" + hashlib.sha256(repr(records).encode("utf-8")).hexdigest()


def _vector_hex(value: object) -> tuple[str, ...]:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (8,) or not np.isfinite(vector).all():
        raise ValueError("grouped central-equivalence vectors must contain eight finite values.")
    return tuple(float(component).hex() for component in vector)


__all__ = [
    "SCHEMA_ID",
    "build_source_equivalence_receipt",
    "grouped_central_equivalence_sha256",
    "verify_source_equivalence_receipt",
]
