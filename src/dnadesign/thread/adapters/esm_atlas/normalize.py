"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/esm_atlas/normalize.py

Normalize ESM Atlas API responses into compact sparse rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dnadesign.thread.adapters.esm_atlas.hashes import raw_response_hash, sequence_md5
from dnadesign.thread.adapters.esm_atlas.models import AtlasNormalizedRows
from dnadesign.thread.foldcheck import sequence_hash as foldcheck_sequence_hash


def normalize_protein_lookup_response(
    *,
    candidate_id: str,
    sequence: str,
    sequence_hash: str,
    response: Mapping[str, Any],
    source_request_hash: str,
    atlas_request_hash: str,
    atlas_query_hash: str,
    atlas_api_base_url: str,
    atlas_api_version: str,
    retrieved_at: str,
    allow_fold_on_miss: bool = False,
) -> AtlasNormalizedRows:
    """Normalize one Atlas /proteins/{hash} response."""

    normalized_sequence = _normalize_sequence(sequence)
    expected_sequence_hash = foldcheck_sequence_hash(normalized_sequence)
    if sequence_hash != expected_sequence_hash:
        raise ValueError(f"sequence_hash mismatch for {candidate_id!r}")
    expected_md5 = sequence_md5(normalized_sequence)
    atlas_hash = _require_text(response, "protein_hash")
    if atlas_hash != expected_md5:
        raise ValueError(f"Atlas protein_hash mismatch for {candidate_id!r}: {atlas_hash} != {expected_md5}")
    if bool(response.get("folded_on_demand", False)) and not allow_fold_on_miss:
        raise ValueError(f"Atlas response for {candidate_id!r} used fold_on_miss without explicit permission")
    observed_length = int(response.get("sequence_length", len(normalized_sequence)))
    if observed_length != len(normalized_sequence):
        raise ValueError(f"Atlas sequence length mismatch for {candidate_id!r}")

    response_hash = raw_response_hash(response)
    features = _feature_rows(response.get("sae_features"))
    top_feature_indices = [int(feature["feature_index"]) for feature in features]
    top_feature_values = [float(feature["value"]) for feature in features]
    top_feature_labels = [str(feature["label"]) for feature in features]
    profile_row = {
        "candidate_id": candidate_id,
        "sequence_hash": sequence_hash,
        "source_request_hash": source_request_hash,
        "atlas_request_hash": atlas_request_hash,
        "atlas_query_hash": atlas_query_hash,
        "atlas_api_base_url": atlas_api_base_url,
        "atlas_api_version": atlas_api_version,
        "query_md5": expected_md5,
        "atlas_hash": atlas_hash,
        "atlas_accession": str(response.get("accession", "")),
        "atlas_source": str(response.get("source", "")),
        "sequence_length": len(normalized_sequence),
        "status": "accepted",
        "folded_on_demand": bool(response.get("folded_on_demand", False)),
        "restricted_count": None,
        "top_feature_indices": top_feature_indices,
        "top_feature_values": top_feature_values,
        "top_feature_labels": top_feature_labels,
        "nearest_hits_json": "[]",
        "raw_response_hash": response_hash,
        "retrieved_at": retrieved_at,
        "failure_reason": "",
    }
    return AtlasNormalizedRows(
        profile_row=profile_row,
        protein_activation_rows=_protein_activation_rows(
            candidate_id=candidate_id,
            sequence_hash=sequence_hash,
            payload=response.get("protein_activations"),
        ),
        residue_activation_rows=_residue_activation_rows(
            candidate_id=candidate_id,
            sequence_hash=sequence_hash,
            payload=response.get("per_residue_activations"),
            expected_residue_count=len(normalized_sequence),
        ),
        feature_catalog_rows=[
            {
                "feature_index": int(feature["feature_index"]),
                "label": str(feature["label"]),
                "description": str(feature.get("description", "")),
                "source_retrieved_at": retrieved_at,
                "raw_feature_hash": raw_response_hash(feature),
            }
            for feature in features
        ],
    )


def build_error_profile_row(
    *,
    candidate_id: str,
    sequence: str,
    sequence_hash: str,
    source_request_hash: str,
    atlas_request_hash: str,
    atlas_query_hash: str,
    atlas_api_base_url: str,
    atlas_api_version: str,
    retrieved_at: str,
    failure_reason: str,
) -> dict[str, object]:
    """Build an explicit errored semantic-profile row without activation data."""

    normalized_sequence = _normalize_sequence(sequence)
    return {
        "candidate_id": candidate_id,
        "sequence_hash": sequence_hash,
        "source_request_hash": source_request_hash,
        "atlas_request_hash": atlas_request_hash,
        "atlas_query_hash": atlas_query_hash,
        "atlas_api_base_url": atlas_api_base_url,
        "atlas_api_version": atlas_api_version,
        "query_md5": sequence_md5(normalized_sequence),
        "atlas_hash": "",
        "atlas_accession": "",
        "atlas_source": "",
        "sequence_length": len(normalized_sequence),
        "status": "errored",
        "folded_on_demand": False,
        "restricted_count": None,
        "top_feature_indices": [],
        "top_feature_values": [],
        "top_feature_labels": [],
        "nearest_hits_json": "[]",
        "raw_response_hash": "",
        "retrieved_at": retrieved_at,
        "failure_reason": failure_reason,
    }


def _feature_rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("Atlas response must include sae_features as a list")
    rows: list[Mapping[str, Any]] = []
    for feature in value:
        if not isinstance(feature, Mapping):
            raise ValueError("Atlas sae_features rows must be mappings")
        if not isinstance(feature.get("feature_index"), int):
            raise ValueError("Atlas feature rows require integer feature_index")
        if "value" not in feature:
            raise ValueError("Atlas feature rows require value")
        if not isinstance(feature.get("label"), str) or not str(feature["label"]).strip():
            raise ValueError("Atlas feature rows require label")
        rows.append(feature)
    return rows


def _protein_activation_rows(
    *,
    candidate_id: str,
    sequence_hash: str,
    payload: Any,
) -> list[dict[str, object]]:
    indices, values = _one_dim_sparse(payload, expected_shape=16384)
    return [
        {
            "candidate_id": candidate_id,
            "sequence_hash": sequence_hash,
            "feature_index": int(feature_index),
            "value": float(value),
        }
        for feature_index, value in zip(indices, values, strict=True)
    ]


def _residue_activation_rows(
    *,
    candidate_id: str,
    sequence_hash: str,
    payload: Any,
    expected_residue_count: int,
) -> list[dict[str, object]]:
    if not isinstance(payload, Mapping):
        raise ValueError("per_residue_activations must be a sparse mapping")
    shape = payload.get("shape")
    if not isinstance(shape, list) or len(shape) != 2 or not all(isinstance(item, int) for item in shape):
        raise ValueError("per_residue_activations requires two-dimensional shape")
    indices = payload.get("indices")
    values = payload.get("values")
    if not isinstance(indices, list) or len(indices) != 2:
        raise ValueError("per_residue_activations requires two sparse index arrays")
    if not isinstance(values, list):
        raise ValueError("per_residue_activations requires values")
    residue_indices = _integer_sequence(indices[0], "per_residue_activations residue indices")
    feature_indices = _integer_sequence(indices[1], "per_residue_activations feature indices")
    activation_values = _float_sequence(values, "per_residue_activations values")
    if len(residue_indices) != len(feature_indices) or len(residue_indices) != len(activation_values):
        raise ValueError("per_residue_activations sparse arrays must have equal length")
    residue_count, feature_count = int(shape[0]), int(shape[1])
    if residue_count != expected_residue_count:
        raise ValueError("per_residue_activations residue count must match sequence length")
    if feature_count != 16384:
        raise ValueError("per_residue_activations must use the 16,384-feature Atlas dictionary")
    rows: list[dict[str, object]] = []
    for residue_index, feature_index, value in zip(residue_indices, feature_indices, activation_values, strict=True):
        if residue_index < 0 or residue_index >= residue_count:
            raise ValueError("per_residue_activations residue index out of bounds")
        if feature_index < 0 or feature_index >= feature_count:
            raise ValueError("per_residue_activations feature index out of bounds")
        rows.append(
            {
                "candidate_id": candidate_id,
                "sequence_hash": sequence_hash,
                "residue_index_zero_based": int(residue_index),
                "sequence_position_one_based": int(residue_index) + 1,
                "feature_index": int(feature_index),
                "value": float(value),
            }
        )
    return rows


def _one_dim_sparse(payload: Any, *, expected_shape: int) -> tuple[list[int], list[float]]:
    if not isinstance(payload, Mapping):
        raise ValueError("protein_activations must be a sparse mapping")
    shape = payload.get("shape")
    if shape != [expected_shape]:
        raise ValueError("protein_activations must use the 16,384-feature Atlas dictionary")
    raw_indices = payload.get("indices")
    raw_values = payload.get("values")
    if not isinstance(raw_indices, list) or not raw_indices:
        raise ValueError("protein_activations requires indices")
    if len(raw_indices) == 1 and isinstance(raw_indices[0], list):
        raw_indices = raw_indices[0]
    indices = _integer_sequence(raw_indices, "protein_activations indices")
    values = _float_sequence(raw_values, "protein_activations values")
    if len(indices) != len(values):
        raise ValueError("protein_activations indices and values must have equal length")
    for feature_index in indices:
        if feature_index < 0 or feature_index >= expected_shape:
            raise ValueError("protein_activations feature index out of bounds")
    return indices, values


def _integer_sequence(value: Any, name: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    result: list[int] = []
    for item in value:
        if not isinstance(item, int):
            raise ValueError(f"{name} must contain integers")
        result.append(int(item))
    return result


def _float_sequence(value: Any, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    result: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)):
            raise ValueError(f"{name} must contain numbers")
        result.append(float(item))
    return result


def _require_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Atlas response requires non-empty {field}")
    return value.strip()


def _normalize_sequence(sequence: str) -> str:
    normalized = "".join(str(sequence).split()).upper()
    if not normalized:
        raise ValueError("sequence must be non-empty")
    return normalized
