"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/sae_window_summary/vectors.py

Vector math for Eco1 SAE window-summary materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.sae_window_summary.constants import (
    INTERPRETATION_LIMIT,
    METHOD_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.sae_window_summary.models import WindowSpec


def build_window_summary_rows(
    *,
    residue_features_path: Path,
    profiles: list[dict[str, Any]],
    window_specs: tuple[WindowSpec, ...],
    design_classes: dict[str, str],
    feature_catalog: dict[tuple[str, int], dict[str, str]],
) -> list[dict[str, Any]]:
    """Build candidate-window rows from sparse residue-feature activations."""

    if not profiles:
        raise ValueError("SAE window summary requires at least one accepted profile row")
    if not any(str(row["candidate_id"]) == "wild_type" for row in profiles):
        raise ValueError("SAE window summary requires candidate_id='wild_type'")
    vectors = _window_vectors(residue_features_path=residue_features_path, window_specs=window_specs)
    rows: list[dict[str, Any]] = []
    for spec in window_specs:
        wt_vector = vectors.get(("wild_type", spec.window_id), {})
        candidate_delta_vectors: dict[str, dict[int, float]] = {}
        for profile in profiles:
            candidate_id = str(profile["candidate_id"])
            vector = vectors.get((candidate_id, spec.window_id), {})
            candidate_delta_vectors[candidate_id] = _delta_vector(vector, wt_vector)
        redundancy = _redundancy(candidate_delta_vectors)
        for profile in profiles:
            candidate_id = str(profile["candidate_id"])
            vector = vectors.get((candidate_id, spec.window_id), {})
            delta = candidate_delta_vectors[candidate_id]
            rows.append(
                _summary_row(
                    profile=profile,
                    spec=spec,
                    vector=vector,
                    wt_vector=wt_vector,
                    delta=delta,
                    design_class_id=design_classes.get(
                        candidate_id, "wild_type_control" if candidate_id == "wild_type" else ""
                    ),
                    feature_catalog=feature_catalog,
                    redundancy=redundancy.get(candidate_id, {}),
                )
            )
    return sorted(rows, key=lambda row: (str(row["window_id"]), str(row["candidate_id"])))


def _window_vectors(
    *,
    residue_features_path: Path,
    window_specs: tuple[WindowSpec, ...],
) -> dict[tuple[str, str], dict[int, float]]:
    window_rows = [
        {"sequence_position_one_based": position, "window_id": spec.window_id}
        for spec in window_specs
        for position in spec.residue_positions_1based
    ]
    if not window_rows:
        raise ValueError("SAE window summary requires at least one residue position")
    window_frame = pl.DataFrame(window_rows)
    grouped = (
        pl.scan_parquet(str(residue_features_path))
        .select(["candidate_id", "sequence_hash", "sae_model", "sequence_position_one_based", "feature_index", "value"])
        .join(window_frame.lazy(), on="sequence_position_one_based", how="inner")
        .group_by(["candidate_id", "window_id", "feature_index"])
        .agg(pl.col("value").sum().alias("activation_sum"))
        .collect()
    )
    vectors: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for row in grouped.iter_rows(named=True):
        vectors[(str(row["candidate_id"]), str(row["window_id"]))][int(row["feature_index"])] = float(
            row["activation_sum"]
        )
    return dict(vectors)


def _summary_row(
    *,
    profile: dict[str, Any],
    spec: WindowSpec,
    vector: dict[int, float],
    wt_vector: dict[int, float],
    delta: dict[int, float],
    design_class_id: str,
    feature_catalog: dict[tuple[str, int], dict[str, str]],
    redundancy: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(profile["candidate_id"])
    sae_model = str(profile["sae_model"])
    return {
        "candidate_id": candidate_id,
        "sequence_hash": str(profile["sequence_hash"]),
        "design_class_id": design_class_id,
        "sae_model": sae_model,
        "model": str(profile["model"]),
        "feature_dictionary_size": int(profile["feature_dictionary_size"]),
        "window_id": spec.window_id,
        "window_label": spec.window_label,
        "residue_count": len(spec.residue_positions_1based),
        "residue_positions_1based_json": json.dumps(list(spec.residue_positions_1based), separators=(",", ":")),
        "window_purpose": spec.purpose,
        "window_vector_hash": _vector_hash(vector),
        "wt_window_vector_hash": _vector_hash(wt_vector),
        "cosine_distance_to_wt": _cosine_distance(vector, wt_vector),
        "activation_sum": float(sum(vector.values())),
        "wt_activation_sum": float(sum(wt_vector.values())),
        "activation_delta_sum_vs_wt": float(sum(delta.values())),
        "top5_signed_feature_deltas_json": _top_delta_json(
            delta=delta,
            vector=vector,
            wt_vector=wt_vector,
            sae_model=sae_model,
            feature_catalog=feature_catalog,
        ),
        "nearest_candidate_id": str(redundancy.get("nearest_candidate_id") or ""),
        "nearest_candidate_window_cosine_distance": redundancy.get("nearest_candidate_window_cosine_distance"),
        "window_redundancy_rank": redundancy.get("window_redundancy_rank"),
        "window_redundancy_group": str(redundancy.get("window_redundancy_group") or ""),
        "window_status": "wt_control" if candidate_id == "wild_type" else "accepted",
        "used_for_selection": False,
        "method_id": METHOD_ID,
        "interpretation_limit": INTERPRETATION_LIMIT,
    }


def _delta_vector(vector: dict[int, float], wt_vector: dict[int, float]) -> dict[int, float]:
    feature_indices = set(vector) | set(wt_vector)
    return {feature: vector.get(feature, 0.0) - wt_vector.get(feature, 0.0) for feature in feature_indices}


def _cosine_distance(left: dict[int, float], right: dict[int, float]) -> float:
    features = set(left) | set(right)
    if not features:
        return 0.0
    dot = sum(left.get(feature, 0.0) * right.get(feature, 0.0) for feature in features)
    left_norm = math.sqrt(sum(left.get(feature, 0.0) ** 2 for feature in features))
    right_norm = math.sqrt(sum(right.get(feature, 0.0) ** 2 for feature in features))
    if left_norm == 0.0 and right_norm == 0.0:
        return 0.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    similarity = max(min(dot / (left_norm * right_norm), 1.0), -1.0)
    return float(1.0 - similarity)


def _top_delta_json(
    *,
    delta: dict[int, float],
    vector: dict[int, float],
    wt_vector: dict[int, float],
    sae_model: str,
    feature_catalog: dict[tuple[str, int], dict[str, str]],
) -> str:
    selected = sorted(delta, key=lambda feature: (abs(delta[feature]), feature), reverse=True)[:5]
    rows = []
    for feature in selected:
        catalog = feature_catalog.get((sae_model, feature), {})
        rows.append(
            {
                "feature_index": feature,
                "activation_delta_vs_wt": float(delta[feature]),
                "candidate_activation": float(vector.get(feature, 0.0)),
                "wt_activation": float(wt_vector.get(feature, 0.0)),
                "label": str(catalog.get("label") or ""),
                "description": str(catalog.get("description") or ""),
            }
        )
    return json.dumps(rows, separators=(",", ":"), sort_keys=True)


def _redundancy(delta_vectors: dict[str, dict[int, float]]) -> dict[str, dict[str, Any]]:
    candidate_ids = [candidate_id for candidate_id in sorted(delta_vectors) if candidate_id != "wild_type"]
    nearest: dict[str, dict[str, Any]] = {"wild_type": {"window_redundancy_group": "wt_control"}}
    if len(candidate_ids) == 1:
        nearest[candidate_ids[0]] = {"window_redundancy_group": "not_available"}
        return nearest
    distance_matrix = _candidate_distance_matrix(
        candidate_ids=candidate_ids,
        vectors={candidate_id: delta_vectors[candidate_id] for candidate_id in candidate_ids},
    )
    distances: list[tuple[str, float]] = []
    for row_index, candidate_id in enumerate(candidate_ids):
        best_index = int(np.argmin(distance_matrix[row_index]))
        best_id = candidate_ids[best_index]
        best_distance = float(distance_matrix[row_index, best_index])
        nearest[candidate_id] = {
            "nearest_candidate_id": best_id,
            "nearest_candidate_window_cosine_distance": best_distance,
            "window_redundancy_group": _redundancy_group(best_distance),
        }
        distances.append((candidate_id, best_distance))
    for rank, (candidate_id, _) in enumerate(sorted(distances, key=lambda item: (item[1], item[0])), start=1):
        nearest[candidate_id]["window_redundancy_rank"] = rank
    return nearest


def _candidate_distance_matrix(
    *,
    candidate_ids: list[str],
    vectors: dict[str, dict[int, float]],
) -> np.ndarray:
    features = sorted({feature for candidate_id in candidate_ids for feature in vectors[candidate_id]})
    if not features:
        distances = np.zeros((len(candidate_ids), len(candidate_ids)), dtype=np.float64)
        np.fill_diagonal(distances, np.inf)
        return distances
    feature_index = {feature: column for column, feature in enumerate(features)}
    matrix = np.zeros((len(candidate_ids), len(features)), dtype=np.float64)
    for row_index, candidate_id in enumerate(candidate_ids):
        for feature, value in vectors[candidate_id].items():
            matrix[row_index, feature_index[feature]] = float(value)
    norms = np.linalg.norm(matrix, axis=1)
    normalized = np.divide(matrix, norms[:, None], out=np.zeros_like(matrix), where=norms[:, None] > 0.0)
    similarities = normalized @ normalized.T
    distances = 1.0 - np.clip(similarities, -1.0, 1.0)
    zero_rows = norms == 0.0
    if np.any(zero_rows):
        distances[zero_rows, :] = 1.0
        distances[:, zero_rows] = 1.0
        zero_indices = np.flatnonzero(zero_rows)
        distances[np.ix_(zero_indices, zero_indices)] = 0.0
    np.fill_diagonal(distances, np.inf)
    return distances


def _redundancy_group(distance: float) -> str:
    if distance <= 0.001:
        return "near_duplicate_window"
    if distance <= 0.05:
        return "close_window"
    return "distinct_window"


def _vector_hash(vector: dict[int, float]) -> str:
    payload = json.dumps(
        [[feature, round(value, 12)] for feature, value in sorted(vector.items())],
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
