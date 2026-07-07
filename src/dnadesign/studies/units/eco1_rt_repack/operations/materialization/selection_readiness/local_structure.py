"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/local_structure.py

Local backbone-shift review metrics for Eco1 RT selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray

from .local_structure_regions import (
    LOCAL_STRUCTURE_REGION_IDS,
    LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_ID,
    LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_NOTE,
    LOCAL_STRUCTURE_RMSD_THRESHOLDS_ANGSTROM,
    LocalStructureRegionSpec,
    local_structure_region_specs,
    position_spec,
)

COORDINATE_SCOPE = "mapped_rt_chain_ca_after_global_fit"
MIN_GLOBAL_ALIGNMENT_CA = 3
MIN_REGION_CA = 3


def mapped_positions_from_residue_map(path: Path) -> list[int]:
    """Return sorted canonical Eco1 positions with mapped backbone coordinates."""

    rows = pq.read_table(path).to_pylist()
    positions = [int(row["canonical_position"]) for row in rows if str(row.get("mapping_status")) == "mapped"]
    if not positions:
        raise ValueError(f"residue map has no mapped canonical positions: {path}")
    return sorted(positions)


def build_local_structure_region_rows(
    *,
    fold_review_rows: Sequence[Mapping[str, Any]],
    reference_backbone_path: Path,
    model_root: Path,
    mapped_positions: Sequence[int],
    contact_geometry_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, object]]:
    """Build candidate-by-region local C-alpha displacement rows."""

    mapped = tuple(sorted({int(position) for position in mapped_positions}))
    region_specs = local_structure_region_specs(
        mapped_positions=mapped,
        contact_geometry_rows=contact_geometry_rows,
    )
    design_class_by_candidate = {
        str(row["candidate_id"]): str(row.get("design_class_id") or "")
        for row in candidate_rows
        if row.get("candidate_id")
    }
    reference_ca = _reference_ca_by_mapped_position(reference_backbone_path, mapped_positions=mapped)
    rows: list[dict[str, object]] = []
    for fold_row in sorted(fold_review_rows, key=lambda row: str(row.get("candidate_id") or "")):
        candidate_id = str(fold_row["candidate_id"])
        design_class_id = str(fold_row.get("design_class_id") or design_class_by_candidate.get(candidate_id, ""))
        source_model_path = _resolve_model_path(
            candidate_id=candidate_id,
            model_artifact_path=_optional_model_path(fold_row.get("model_artifact_path")),
            model_root=model_root,
        )
        if reference_ca is None:
            rows.extend(
                _status_rows(
                    candidate_id=candidate_id,
                    design_class_id=design_class_id,
                    region_specs=region_specs,
                    status="reference_structure_missing",
                    status_reason=str(reference_backbone_path),
                    source_model_path=source_model_path,
                    reference_model_path=reference_backbone_path,
                )
            )
            continue
        if source_model_path is None:
            rows.extend(
                _status_rows(
                    candidate_id=candidate_id,
                    design_class_id=design_class_id,
                    region_specs=region_specs,
                    status="model_structure_missing",
                    status_reason=str(
                        _optional_model_path(fold_row.get("model_artifact_path")) or model_root / f"{candidate_id}.pdb"
                    ),
                    source_model_path=None,
                    reference_model_path=reference_backbone_path,
                )
            )
            continue
        candidate_ca = _candidate_ca_by_mapped_position(source_model_path, mapped_positions=mapped)
        shared_global = [position for position in mapped if position in reference_ca and position in candidate_ca]
        if len(shared_global) < MIN_GLOBAL_ALIGNMENT_CA:
            rows.extend(
                _status_rows(
                    candidate_id=candidate_id,
                    design_class_id=design_class_id,
                    region_specs=region_specs,
                    status="insufficient_alignment_overlap",
                    status_reason=(
                        f"shared mapped C-alpha count {len(shared_global)} is below {MIN_GLOBAL_ALIGNMENT_CA}"
                    ),
                    source_model_path=source_model_path,
                    reference_model_path=reference_backbone_path,
                    reference_ca=reference_ca,
                    candidate_ca=candidate_ca,
                )
            )
            continue
        aligned_candidate = _aligned_candidate_ca(
            candidate_ca=candidate_ca,
            reference_ca=reference_ca,
            shared_positions=shared_global,
        )
        for spec in region_specs:
            rows.append(
                _metric_row(
                    candidate_id=candidate_id,
                    design_class_id=design_class_id,
                    spec=spec,
                    reference_ca=reference_ca,
                    candidate_ca=aligned_candidate,
                    source_model_path=source_model_path,
                    reference_model_path=reference_backbone_path,
                )
            )
    return rows


def build_local_structure_review_by_candidate(
    local_structure_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, object]]:
    """Summarize local-structure rows into a candidate-level gate contract."""

    required_regions = set(LOCAL_STRUCTURE_REGION_IDS)
    rows_by_candidate: dict[str, list[Mapping[str, Any]]] = {}
    for row in local_structure_rows:
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id:
            rows_by_candidate.setdefault(candidate_id, []).append(row)
    summaries: dict[str, dict[str, object]] = {}
    for candidate_id, rows in rows_by_candidate.items():
        rows_by_region = {str(row.get("region_id") or ""): row for row in rows}
        missing_regions = sorted(required_regions - set(rows_by_region))
        unavailable_reasons = [f"{region_id}:missing_metric_row" for region_id in missing_regions]
        available_values: list[float] = []
        threshold_failures: list[str] = []
        per_region_fields: dict[str, object] = {}
        for region_id in LOCAL_STRUCTURE_REGION_IDS:
            row = rows_by_region.get(region_id)
            status = "" if row is None else str(row.get("status") or "")
            value = None if row is None else row.get("local_ca_rmsd_angstrom")
            per_region_fields[f"local_structure_{region_id}_ca_rmsd_angstrom"] = None if value is None else float(value)
            if status == "available" and value is not None:
                numeric_value = float(value)
                available_values.append(numeric_value)
                threshold = float(
                    (row or {}).get("local_ca_rmsd_threshold_angstrom")
                    or LOCAL_STRUCTURE_RMSD_THRESHOLDS_ANGSTROM[region_id]
                )
                if (
                    str(row.get("local_ca_rmsd_threshold_status") or "") == "threshold_exceeded"
                    or numeric_value > threshold
                ):
                    threshold_failures.append(f"{region_id}:local_ca_rmsd {numeric_value:.3f} > {threshold:.3f}")
                continue
            if row is not None:
                unavailable_reasons.append(f"{region_id}:{status or 'missing_status'}")
        if unavailable_reasons:
            gate_status = "unavailable"
        elif threshold_failures:
            gate_status = "threshold_exceeded"
        else:
            gate_status = "passed"
        summaries[candidate_id] = {
            "local_structure_gate_status": gate_status,
            "local_structure_gate_failure_reasons_json": json.dumps(
                sorted([*unavailable_reasons, *threshold_failures])
            ),
            "local_structure_region_count": len(LOCAL_STRUCTURE_REGION_IDS),
            "local_structure_available_region_count": len(available_values),
            "local_structure_unavailable_region_count": len(unavailable_reasons),
            "local_structure_threshold_failed_region_count": len(threshold_failures),
            "local_structure_threshold_policy_id": LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_ID,
            "local_structure_max_ca_rmsd_angstrom": round(max(available_values), 3) if available_values else None,
            "local_structure_mean_ca_rmsd_angstrom": round(sum(available_values) / len(available_values), 3)
            if available_values
            else None,
            **per_region_fields,
        }
    return summaries


def _metric_row(
    *,
    candidate_id: str,
    design_class_id: str,
    spec: LocalStructureRegionSpec,
    reference_ca: Mapping[int, NDArray[np.float64]],
    candidate_ca: Mapping[int, NDArray[np.float64]],
    source_model_path: Path,
    reference_model_path: Path,
) -> dict[str, object]:
    reference_positions = [position for position in spec.positions if position in reference_ca]
    candidate_positions = [position for position in spec.positions if position in candidate_ca]
    shared_positions = [
        position for position in spec.positions if position in reference_ca and position in candidate_ca
    ]
    base = _base_row(
        candidate_id=candidate_id,
        design_class_id=design_class_id,
        spec=spec,
        n_reference_ca=len(reference_positions),
        n_candidate_ca=len(candidate_positions),
        n_shared_ca=len(shared_positions),
        source_model_path=source_model_path,
        reference_model_path=reference_model_path,
    )
    if len(shared_positions) < MIN_REGION_CA:
        return {
            **base,
            "local_ca_rmsd_angstrom": None,
            "mean_ca_displacement_angstrom": None,
            "max_ca_displacement_angstrom": None,
            "local_ca_rmsd_threshold_status": "not_evaluated",
            "status": "insufficient_region_overlap",
            "status_reason": f"shared region C-alpha count {len(shared_positions)} is below {MIN_REGION_CA}",
        }
    displacements = [
        float(np.linalg.norm(candidate_ca[position] - reference_ca[position])) for position in shared_positions
    ]
    squared = sum(displacement * displacement for displacement in displacements) / len(displacements)
    local_ca_rmsd = round(math.sqrt(squared), 3)
    return {
        **base,
        "local_ca_rmsd_angstrom": local_ca_rmsd,
        "mean_ca_displacement_angstrom": round(sum(displacements) / len(displacements), 3),
        "max_ca_displacement_angstrom": round(max(displacements), 3),
        "local_ca_rmsd_threshold_status": _threshold_status(
            region_id=spec.region_id,
            local_ca_rmsd=local_ca_rmsd,
        ),
        "status": "available",
        "status_reason": "",
    }


def _status_rows(
    *,
    candidate_id: str,
    design_class_id: str,
    region_specs: Sequence[LocalStructureRegionSpec],
    status: str,
    status_reason: str,
    source_model_path: Path | None,
    reference_model_path: Path,
    reference_ca: Mapping[int, NDArray[np.float64]] | None = None,
    candidate_ca: Mapping[int, NDArray[np.float64]] | None = None,
) -> list[dict[str, object]]:
    return [
        {
            **_base_row(
                candidate_id=candidate_id,
                design_class_id=design_class_id,
                spec=spec,
                n_reference_ca=len(
                    [position for position in spec.positions if reference_ca and position in reference_ca]
                ),
                n_candidate_ca=len(
                    [position for position in spec.positions if candidate_ca and position in candidate_ca]
                ),
                n_shared_ca=len(
                    [
                        position
                        for position in spec.positions
                        if reference_ca and candidate_ca and position in reference_ca and position in candidate_ca
                    ]
                ),
                source_model_path=source_model_path,
                reference_model_path=reference_model_path,
            ),
            "local_ca_rmsd_angstrom": None,
            "mean_ca_displacement_angstrom": None,
            "max_ca_displacement_angstrom": None,
            "local_ca_rmsd_threshold_status": "not_evaluated",
            "status": status,
            "status_reason": status_reason,
        }
        for spec in region_specs
    ]


def _base_row(
    *,
    candidate_id: str,
    design_class_id: str,
    spec: LocalStructureRegionSpec,
    n_reference_ca: int,
    n_candidate_ca: int,
    n_shared_ca: int,
    source_model_path: Path | None,
    reference_model_path: Path,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "design_class_id": design_class_id,
        "region_id": spec.region_id,
        "region_label": spec.label,
        "region_role": spec.role,
        "region_position_count": len(spec.positions),
        "region_position_spec": position_spec(spec.positions),
        "region_position_source": spec.position_source,
        "region_source_basis_ids_json": json.dumps(list(spec.source_basis_ids), sort_keys=True),
        "coordinate_scope": COORDINATE_SCOPE,
        "n_reference_ca": n_reference_ca,
        "n_candidate_ca": n_candidate_ca,
        "n_shared_ca": n_shared_ca,
        "local_ca_rmsd_threshold_angstrom": LOCAL_STRUCTURE_RMSD_THRESHOLDS_ANGSTROM[spec.region_id],
        "local_ca_rmsd_threshold_policy_id": LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_ID,
        "local_ca_rmsd_threshold_policy_note": LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_NOTE,
        "source_model_path": "" if source_model_path is None else str(source_model_path),
        "reference_model_path": str(reference_model_path),
    }


def _threshold_status(*, region_id: str, local_ca_rmsd: float) -> str:
    threshold = LOCAL_STRUCTURE_RMSD_THRESHOLDS_ANGSTROM[region_id]
    return "passed" if float(local_ca_rmsd) <= threshold else "threshold_exceeded"


def _aligned_candidate_ca(
    *,
    candidate_ca: Mapping[int, NDArray[np.float64]],
    reference_ca: Mapping[int, NDArray[np.float64]],
    shared_positions: Sequence[int],
) -> dict[int, NDArray[np.float64]]:
    candidate_coords = np.asarray([candidate_ca[position] for position in shared_positions], dtype=float)
    reference_coords = np.asarray([reference_ca[position] for position in shared_positions], dtype=float)
    rotation, candidate_centroid, reference_centroid = _kabsch(candidate_coords, reference_coords)
    return {
        position: (coord - candidate_centroid) @ rotation + reference_centroid
        for position, coord in candidate_ca.items()
    }


def _kabsch(
    candidate_coords: NDArray[np.float64],
    reference_coords: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    candidate_centroid = candidate_coords.mean(axis=0)
    reference_centroid = reference_coords.mean(axis=0)
    candidate_centered = candidate_coords - candidate_centroid
    reference_centered = reference_coords - reference_centroid
    covariance = candidate_centered.T @ reference_centered
    left, _singular_values, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(left @ right_t))
    return left @ correction @ right_t, candidate_centroid, reference_centroid


def _reference_ca_by_mapped_position(
    path: Path, *, mapped_positions: Sequence[int]
) -> dict[int, NDArray[np.float64]] | None:
    if not path.exists():
        return None
    by_residue, ordered = _ca_coordinates(path)
    mapped = tuple(int(position) for position in mapped_positions)
    if all(position in by_residue for position in mapped):
        return {position: by_residue[position] for position in mapped}
    if len(ordered) == len(mapped):
        return {position: coord for position, coord in zip(mapped, ordered, strict=True)}
    return {position: by_residue[position] for position in mapped if position in by_residue}


def _candidate_ca_by_mapped_position(path: Path, *, mapped_positions: Sequence[int]) -> dict[int, NDArray[np.float64]]:
    by_residue, ordered = _ca_coordinates(path)
    mapped = tuple(int(position) for position in mapped_positions)
    if all(position in by_residue for position in mapped):
        return {position: by_residue[position] for position in mapped}
    if ordered and len(ordered) >= max(mapped):
        return {position: ordered[position - 1] for position in mapped if position <= len(ordered)}
    if len(ordered) == len(mapped):
        return {position: coord for position, coord in zip(mapped, ordered, strict=True)}
    return {position: by_residue[position] for position in mapped if position in by_residue}


def _ca_coordinates(path: Path) -> tuple[dict[int, NDArray[np.float64]], list[NDArray[np.float64]]]:
    by_residue: dict[int, NDArray[np.float64]] = {}
    ordered: list[NDArray[np.float64]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or line[12:16].strip() != "CA":
            continue
        coord = np.asarray([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float)
        ordered.append(coord)
        by_residue[int(line[22:26])] = coord
    return by_residue, ordered


def _optional_model_path(value: object) -> Path | None:
    text = "" if value is None else str(value).strip()
    return None if not text else Path(text)


def _resolve_model_path(*, candidate_id: str, model_artifact_path: Path | None, model_root: Path) -> Path | None:
    if model_artifact_path is not None and model_artifact_path.exists():
        return model_artifact_path
    if model_artifact_path is not None and (model_root / model_artifact_path.name).exists():
        return model_root / model_artifact_path.name
    candidate_named = model_root / f"{candidate_id}.pdb"
    return candidate_named if candidate_named.exists() else None


__all__ = [
    "COORDINATE_SCOPE",
    "LOCAL_STRUCTURE_REGION_IDS",
    "LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_ID",
    "LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_NOTE",
    "LOCAL_STRUCTURE_RMSD_THRESHOLDS_ANGSTROM",
    "LocalStructureRegionSpec",
    "build_local_structure_region_rows",
    "build_local_structure_review_by_candidate",
    "local_structure_region_specs",
    "mapped_positions_from_residue_map",
]
