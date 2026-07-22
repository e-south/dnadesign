"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/biohub_esmc_sae_audit.py

Biohub ESMC SAE provenance audit helpers for Eco1 review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def sae_provenance_audit(
    *,
    profile_path: Path,
    protein_features_path: Path,
    residue_features_path: Path,
) -> dict[str, Any]:
    """Summarize SAE row coverage and WT-vector similarity provenance."""

    profile_rows = pq.read_table(
        profile_path,
        columns=_available_columns(
            profile_path,
            [
                "candidate_id",
                "sequence_hash",
                "raw_logits_response_hash",
                "sequence_length",
                "status",
            ],
        ),
    ).to_pylist()
    accepted_rows = [row for row in profile_rows if str(row.get("status") or "") == "accepted"]
    sequence_lengths = [int(row["sequence_length"]) for row in profile_rows if row.get("sequence_length") is not None]
    activation_cosines = _wt_activation_cosines(protein_features_path)
    return {
        "profile_row_count": pq.read_metadata(profile_path).num_rows,
        "accepted_profile_count": len(accepted_rows),
        "unique_sequence_hash_count": _unique_count(row.get("sequence_hash") for row in accepted_rows),
        "unique_raw_logits_response_hash_count": _unique_count(
            row.get("raw_logits_response_hash") for row in accepted_rows
        ),
        "protein_feature_row_count": pq.read_metadata(protein_features_path).num_rows,
        "residue_feature_row_count": pq.read_metadata(residue_features_path).num_rows,
        "sequence_length_min": min(sequence_lengths) if sequence_lengths else 0,
        "sequence_length_max": max(sequence_lengths) if sequence_lengths else 0,
        "activation_similarity_basis": "cosine_similarity_of_activation_sum_vectors_against_wt",
        "wt_activation_cosine_min": min(activation_cosines) if activation_cosines else None,
        "wt_activation_cosine_max": max(activation_cosines) if activation_cosines else None,
        "wt_activation_cosine_mean": sum(activation_cosines) / len(activation_cosines) if activation_cosines else None,
    }


def _available_columns(path: Path, columns: list[str]) -> list[str]:
    names = set(pq.read_schema(path).names)
    return [column for column in columns if column in names]


def _unique_count(values: Any) -> int:
    return len({str(value) for value in values if value})


def _wt_activation_cosines(protein_features_path: Path) -> list[float]:
    rows = pq.read_table(
        protein_features_path,
        columns=["candidate_id", "feature_index", "activation_sum"],
    ).to_pylist()
    vectors: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id:
            vectors[candidate_id][int(row["feature_index"])] = float(row.get("activation_sum") or 0.0)
    wt_vector = vectors.get("wild_type") or {}
    if not wt_vector:
        return []
    return [
        _cosine_similarity(wt_vector, vector) for candidate_id, vector in vectors.items() if candidate_id != "wild_type"
    ]


def _cosine_similarity(left: dict[int, float], right: dict[int, float]) -> float:
    feature_indices = set(left) | set(right)
    numerator = sum(left.get(feature_index, 0.0) * right.get(feature_index, 0.0) for feature_index in feature_indices)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)
