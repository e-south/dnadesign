"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/biohub_esmc/normalize.py

Normalize Biohub ESMC logits responses into compact SAE tables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from dnadesign.thread.adapters.biohub_esmc.encoded import SparseSaeTensor, decode_sae_outputs
from dnadesign.thread.adapters.biohub_esmc.hashes import raw_response_hash
from dnadesign.thread.adapters.biohub_esmc.models import BiohubEsmcNormalizedRows
from dnadesign.thread.foldcheck import sequence_hash as foldcheck_sequence_hash


def normalize_logits_response(
    *,
    candidate_id: str,
    sequence: str,
    sequence_hash: str,
    encode_response: Mapping[str, Any],
    logits_response: Mapping[str, Any],
    source_request_hash: str,
    biohub_request_hash: str,
    biohub_query_hash: str,
    biohub_api_base_url: str,
    biohub_api_version: str,
    model: str,
    sae_model: str,
    normalize_features: bool,
    key_label: str,
    retrieved_at: str,
) -> BiohubEsmcNormalizedRows:
    """Normalize one authenticated Biohub ESMC logits response."""

    normalized_sequence = _normalize_sequence(sequence)
    expected_sequence_hash = foldcheck_sequence_hash(normalized_sequence)
    if sequence_hash != expected_sequence_hash:
        raise ValueError(f"sequence_hash mismatch for {candidate_id!r}")
    if not isinstance(logits_response, Mapping):
        raise ValueError("Biohub logits response must be a mapping")
    sparse = decode_sae_outputs(
        logits_response.get("sae_outputs"),
        sequence_length=len(normalized_sequence),
        sae_model=sae_model,
    )
    residue_rows = _residue_feature_rows(
        candidate_id=candidate_id,
        sequence_hash=sequence_hash,
        sparse=sparse,
    )
    protein_rows = _protein_feature_rows(
        candidate_id=candidate_id,
        sequence_hash=sequence_hash,
        sparse=sparse,
    )
    encode_hash = raw_response_hash(encode_response)
    logits_hash = raw_response_hash(logits_response)
    profile_row = {
        "candidate_id": candidate_id,
        "sequence_hash": sequence_hash,
        "source_request_hash": source_request_hash,
        "biohub_request_hash": biohub_request_hash,
        "biohub_query_hash": biohub_query_hash,
        "biohub_api_base_url": biohub_api_base_url,
        "biohub_api_version": biohub_api_version,
        "model": model,
        "sae_model": sae_model,
        "normalize_features": bool(normalize_features),
        "key_label": key_label,
        "sequence_length": len(normalized_sequence),
        "token_count": sparse.token_count,
        "feature_dictionary_size": sparse.feature_count,
        "status": "accepted",
        "protein_feature_count": len(protein_rows),
        "residue_feature_count": len(residue_rows),
        "encoded_sae_bytes": sparse.encoded_sae_bytes,
        "raw_encode_response_hash": encode_hash,
        "raw_logits_response_hash": logits_hash,
        "retrieved_at": retrieved_at,
        "failure_reason": "",
    }
    return BiohubEsmcNormalizedRows(
        profile_row=profile_row,
        protein_feature_rows=protein_rows,
        residue_feature_rows=residue_rows,
        feature_catalog_rows=_feature_catalog_rows(
            feature_indices={int(row["feature_index"]) for row in protein_rows},
            sae_model=sae_model,
            retrieved_at=retrieved_at,
        ),
    )


def build_error_profile_row(
    *,
    candidate_id: str,
    sequence: str,
    sequence_hash: str,
    source_request_hash: str,
    biohub_request_hash: str,
    biohub_query_hash: str,
    biohub_api_base_url: str,
    biohub_api_version: str,
    model: str,
    sae_model: str,
    normalize_features: bool,
    key_label: str,
    retrieved_at: str,
    failure_reason: str,
) -> dict[str, object]:
    """Build an explicit errored Biohub ESMC profile row."""

    normalized_sequence = _normalize_sequence(sequence)
    return {
        "candidate_id": candidate_id,
        "sequence_hash": sequence_hash,
        "source_request_hash": source_request_hash,
        "biohub_request_hash": biohub_request_hash,
        "biohub_query_hash": biohub_query_hash,
        "biohub_api_base_url": biohub_api_base_url,
        "biohub_api_version": biohub_api_version,
        "model": model,
        "sae_model": sae_model,
        "normalize_features": bool(normalize_features),
        "key_label": key_label,
        "sequence_length": len(normalized_sequence),
        "token_count": None,
        "feature_dictionary_size": None,
        "status": "errored",
        "protein_feature_count": 0,
        "residue_feature_count": 0,
        "encoded_sae_bytes": None,
        "raw_encode_response_hash": "",
        "raw_logits_response_hash": "",
        "retrieved_at": retrieved_at,
        "failure_reason": _compact_failure_reason(failure_reason),
    }


def _residue_feature_rows(
    *,
    candidate_id: str,
    sequence_hash: str,
    sparse: SparseSaeTensor,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for residue_index, feature_index, value in zip(
        sparse.residue_indices,
        sparse.feature_indices,
        sparse.values,
        strict=True,
    ):
        rows.append(
            {
                "candidate_id": candidate_id,
                "sequence_hash": sequence_hash,
                "sae_model": sparse.sae_model,
                "residue_index_zero_based": int(residue_index),
                "sequence_position_one_based": int(residue_index) + 1,
                "feature_index": int(feature_index),
                "value": float(value),
            }
        )
    return rows


def _protein_feature_rows(
    *,
    candidate_id: str,
    sequence_hash: str,
    sparse: SparseSaeTensor,
) -> list[dict[str, object]]:
    grouped_values: dict[int, list[float]] = defaultdict(list)
    grouped_residues: dict[int, set[int]] = defaultdict(set)
    for residue_index, feature_index, value in zip(
        sparse.residue_indices,
        sparse.feature_indices,
        sparse.values,
        strict=True,
    ):
        grouped_values[int(feature_index)].append(float(value))
        grouped_residues[int(feature_index)].add(int(residue_index))
    rows: list[dict[str, object]] = []
    for feature_index in sorted(grouped_values):
        values = grouped_values[feature_index]
        activation_sum = float(sum(values))
        rows.append(
            {
                "candidate_id": candidate_id,
                "sequence_hash": sequence_hash,
                "sae_model": sparse.sae_model,
                "feature_index": int(feature_index),
                "sequence_residue_count": sparse.sequence_residue_count,
                "nonzero_residue_count": len(grouped_residues[feature_index]),
                "activation_sum": activation_sum,
                "activation_mean": activation_sum / sparse.sequence_residue_count,
                "activation_max": max(values),
            }
        )
    return rows


def _feature_catalog_rows(
    *,
    feature_indices: set[int],
    sae_model: str,
    retrieved_at: str,
) -> list[dict[str, object]]:
    return [
        {
            "sae_model": sae_model,
            "feature_index": int(feature_index),
            "label": "",
            "description": "",
            "source_retrieved_at": retrieved_at,
            "raw_feature_hash": "",
        }
        for feature_index in sorted(feature_indices)
    ]


def _normalize_sequence(sequence: str) -> str:
    normalized = "".join(str(sequence).split()).upper()
    if not normalized:
        raise ValueError("sequence must be non-empty")
    return normalized


def _compact_failure_reason(reason: str, *, max_length: int = 500) -> str:
    compact = " ".join(str(reason).split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3] + "..."
