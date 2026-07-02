"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/biohub_esmc_sae_tables.py

Compact tables for Biohub ESMC SAE review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)

PER_PROTEIN_TOP_FEATURE_COUNT = 8
FEATURE_PREVALENCE_THRESHOLD = 0.01
_PROTEIN_FEATURE_COLUMNS = [
    "candidate_id",
    "sequence_hash",
    "sae_model",
    "feature_index",
    "sequence_residue_count",
    "nonzero_residue_count",
    "activation_sum",
    "activation_mean",
    "activation_max",
]
_FEATURE_CATALOG_COLUMNS = ["sae_model", "feature_index", "label", "description"]


def write_protein_top_feature_table(
    *,
    path: Path,
    protein_features_path: Path,
    residue_features_path: Path,
    feature_catalog_path: Path,
    top_n: int = PER_PROTEIN_TOP_FEATURE_COUNT,
) -> None:
    """Write per-protein top SAE features by peak activation and thresholded prevalence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    feature_rows = pq.read_table(protein_features_path, columns=_PROTEIN_FEATURE_COLUMNS).to_pylist()
    thresholded = thresholded_feature_stats(residue_features_path)
    descriptions = _feature_descriptions(feature_catalog_path)
    rows: list[dict[str, Any]] = []
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in feature_rows:
        by_candidate.setdefault(str(row["candidate_id"]), []).append(dict(row))
    for candidate_id in sorted(by_candidate):
        rows.extend(
            _candidate_top_feature_rows(
                candidate_id=candidate_id,
                candidate_rows=by_candidate[candidate_id],
                thresholded=thresholded,
                descriptions=descriptions,
                top_n=top_n,
            )
        )
    pq.write_table(pa.Table.from_pylist(rows), path)


def make_protein_top_feature_table_row(
    *,
    table_path: Path,
    protein_features_path: Path,
    residue_features_path: Path,
    feature_catalog_path: Path,
    request_manifest_path: Path,
    section: str,
    source_tables: list[str],
    interpretation_limit: str,
    method_summary: str,
    source_notebook: str,
) -> dict[str, Any]:
    """Build the manifest row for the per-protein top-feature table."""

    return make_deliverable_row(
        deliverable_id="biohub_esmc_protein_top_sae_features",
        section=section,
        artifact_kind="parquet",
        status="materialized",
        path=table_path,
        source_tables=source_tables,
        input_hashes=file_hashes(
            {
                "protein_features": protein_features_path,
                "residue_features": residue_features_path,
                "feature_catalog": feature_catalog_path,
                "request_manifest": request_manifest_path,
            }
        ),
        alt_text=(
            "Compact table of per-protein top Biohub ESMC SAE features ordered by peak activation "
            "and activation-thresholded prevalence."
        ),
        description=(
            "Supports per-protein feature inspection. Feature descriptions are populated only "
            "when the exact SAE dictionary supplies source-backed labels."
        ),
        interpretation_limit=interpretation_limit,
        title="Per-protein top SAE features are ordered by peak activation and thresholded prevalence",
        method_summary=method_summary,
        evidence_summary={
            "source_notebook": source_notebook,
            "top_features_per_metric": PER_PROTEIN_TOP_FEATURE_COUNT,
            "prevalence_activation_threshold": FEATURE_PREVALENCE_THRESHOLD,
            "description_policy": "exact_sae_dictionary_only",
        },
        role="review_only",
    )


def _candidate_top_feature_rows(
    *,
    candidate_id: str,
    candidate_rows: list[dict[str, Any]],
    thresholded: dict[tuple[str, int], dict[str, float | int]],
    descriptions: dict[tuple[str, int], tuple[str, str, str]],
    top_n: int,
) -> list[dict[str, Any]]:
    by_max = sorted(
        candidate_rows,
        key=lambda row: (
            float(row["activation_max"]),
            int(row["nonzero_residue_count"]),
            float(row["activation_sum"]),
        ),
        reverse=True,
    )[:top_n]
    by_prevalence = sorted(
        candidate_rows,
        key=lambda row: (
            int(thresholded_stats(row, thresholded)["prevalent_residue_count"]),
            float(row["activation_max"]),
            float(thresholded_stats(row, thresholded)["prevalent_activation_sum"]),
        ),
        reverse=True,
    )[:top_n]
    ranks_by_max = {int(row["feature_index"]): rank for rank, row in enumerate(by_max, start=1)}
    ranks_by_prevalence = {int(row["feature_index"]): rank for rank, row in enumerate(by_prevalence, start=1)}
    selected = {int(row["feature_index"]) for row in [*by_max, *by_prevalence]}
    indexed_rows = {int(row["feature_index"]): row for row in candidate_rows}
    return [
        _top_feature_row(
            candidate_id=candidate_id,
            row=indexed_rows[feature_index],
            feature_index=feature_index,
            ranks_by_max=ranks_by_max,
            ranks_by_prevalence=ranks_by_prevalence,
            thresholded=thresholded,
            descriptions=descriptions,
        )
        for feature_index in sorted(selected)
    ]


def _top_feature_row(
    *,
    candidate_id: str,
    row: dict[str, Any],
    feature_index: int,
    ranks_by_max: dict[int, int],
    ranks_by_prevalence: dict[int, int],
    thresholded: dict[tuple[str, int], dict[str, float | int]],
    descriptions: dict[tuple[str, int], tuple[str, str, str]],
) -> dict[str, Any]:
    sequence_residue_count = int(row["sequence_residue_count"])
    stats = thresholded_stats(row, thresholded)
    prevalent_residue_count = int(stats["prevalent_residue_count"])
    label, description, description_status = descriptions.get(
        (str(row["sae_model"]), feature_index),
        ("", "", "not_available_exact_dictionary_unlabeled"),
    )
    return {
        "candidate_id": candidate_id,
        "sequence_hash": str(row["sequence_hash"]),
        "sae_model": str(row["sae_model"]),
        "feature_index": feature_index,
        "rank_by_max_activation": ranks_by_max.get(feature_index),
        "rank_by_prevalence": ranks_by_prevalence.get(feature_index),
        "selection_reason": _selection_reason(
            feature_index=feature_index,
            ranks_by_max=ranks_by_max,
            ranks_by_prevalence=ranks_by_prevalence,
        ),
        "sequence_residue_count": sequence_residue_count,
        "nonzero_residue_count": int(row["nonzero_residue_count"]),
        "prevalence_activation_threshold": FEATURE_PREVALENCE_THRESHOLD,
        "prevalent_residue_count": prevalent_residue_count,
        "prevalence_fraction": prevalent_residue_count / sequence_residue_count,
        "mean_prevalent_activation": float(stats["mean_prevalent_activation"]),
        "activation_sum": float(row["activation_sum"]),
        "activation_mean": float(row["activation_mean"]),
        "activation_max": float(row["activation_max"]),
        "label": label,
        "description": description,
        "description_status": description_status,
        "description_source": "biohub_esmc_feature_catalog.parquet" if description else "",
    }


def thresholded_feature_stats(path: Path) -> dict[tuple[str, int], dict[str, float | int]]:
    """Summarize tutorial-aligned SAE feature prevalence above the activation threshold."""

    table = pq.read_table(path, columns=["candidate_id", "feature_index", "value"])
    filtered = table.filter(pc.greater(table.column("value"), FEATURE_PREVALENCE_THRESHOLD))
    if filtered.num_rows == 0:
        return {}
    grouped = filtered.group_by(["candidate_id", "feature_index"]).aggregate([("value", "count"), ("value", "sum")])
    stats: dict[tuple[str, int], dict[str, float | int]] = {}
    for row in grouped.to_pylist():
        count = int(row["value_count"])
        activation_sum = float(row["value_sum"])
        stats[(str(row["candidate_id"]), int(row["feature_index"]))] = {
            "prevalent_residue_count": count,
            "prevalent_activation_sum": activation_sum,
            "mean_prevalent_activation": activation_sum / count if count else 0.0,
        }
    return stats


def thresholded_stats(
    row: dict[str, Any],
    thresholded: dict[tuple[str, int], dict[str, float | int]],
) -> dict[str, float | int]:
    return thresholded.get(
        (str(row["candidate_id"]), int(row["feature_index"])),
        {
            "prevalent_residue_count": 0,
            "prevalent_activation_sum": 0.0,
            "mean_prevalent_activation": 0.0,
        },
    )


def _feature_descriptions(path: Path) -> dict[tuple[str, int], tuple[str, str, str]]:
    descriptions: dict[tuple[str, int], tuple[str, str, str]] = {}
    for row in pq.read_table(path, columns=_FEATURE_CATALOG_COLUMNS).to_pylist():
        label = str(row.get("label") or "")
        description = str(row.get("description") or "")
        status = (
            "source_backed_exact_dictionary_description"
            if label or description
            else "not_available_exact_dictionary_unlabeled"
        )
        descriptions[(str(row["sae_model"]), int(row["feature_index"]))] = (label, description, status)
    return descriptions


def _selection_reason(
    *,
    feature_index: int,
    ranks_by_max: dict[int, int],
    ranks_by_prevalence: dict[int, int],
) -> str:
    in_max = feature_index in ranks_by_max
    in_prevalence = feature_index in ranks_by_prevalence
    if in_max and in_prevalence:
        return "top_by_max_activation_and_prevalence"
    if in_max:
        return "top_by_max_activation"
    return "top_by_prevalence"
