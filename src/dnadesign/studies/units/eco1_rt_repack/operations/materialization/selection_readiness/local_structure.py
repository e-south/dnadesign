"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/local_structure.py

Local backbone-shift review metrics for Eco1 RT selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.review_axes import (
    DIRECT_CONTACT_DISTANCE_ANGSTROM,
    NA_FACING_DISTANCE_ANGSTROM,
    WANG_THUMB_CONTACT_TRACK_POSITIONS,
)

COORDINATE_SCOPE = "mapped_rt_chain_ca_after_global_fit"
MIN_GLOBAL_ALIGNMENT_CA = 3
MIN_REGION_CA = 3


@dataclass(frozen=True)
class LocalStructureRegionSpec:
    """One named Eco1 RT region for local backbone-shift review."""

    region_id: str
    label: str
    role: str
    positions: tuple[int, ...]


_STATIC_REGION_SPECS = (
    LocalStructureRegionSpec(
        region_id="catalytic_initiation_context",
        label="Catalytic YADD context",
        role="catalytic_initiation_review",
        positions=tuple(range(189, 205)),
    ),
    LocalStructureRegionSpec(
        region_id="retron_x_naxxh_context",
        label="Retron X NAxxH context",
        role="retron_motif_review",
        positions=tuple(range(99, 116)),
    ),
    LocalStructureRegionSpec(
        region_id="retron_y_vtg_context",
        label="Retron Y VTG context",
        role="retron_motif_review",
        positions=tuple(range(237, 252)),
    ),
    LocalStructureRegionSpec(
        region_id="thumb_contact_track_context",
        label="Wang thumb-contact track",
        role="thumb_contact_review",
        positions=tuple(sorted(WANG_THUMB_CONTACT_TRACK_POSITIONS)),
    ),
)

LOCAL_STRUCTURE_REGION_IDS = tuple(
    spec.region_id
    for spec in (
        *_STATIC_REGION_SPECS,
        LocalStructureRegionSpec(
            region_id="near_retained_dna_rna_annulus",
            label="Near retained DNA/RNA annulus",
            role="substrate_proximal_review",
            positions=(),
        ),
        LocalStructureRegionSpec(
            region_id="distal_scaffold_control",
            label="Distal scaffold control",
            role="distal_scaffold_control",
            positions=(),
        ),
    )
)


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


def local_structure_region_specs(
    *,
    mapped_positions: Sequence[int],
    contact_geometry_rows: Sequence[Mapping[str, Any]],
) -> tuple[LocalStructureRegionSpec, ...]:
    """Return static and derived Eco1 local-structure regions."""

    mapped = set(int(position) for position in mapped_positions)
    static_positions = {position for spec in _STATIC_REGION_SPECS for position in spec.positions}
    direct_contact_positions: set[int] = set()
    annulus_positions: set[int] = set()
    for row in contact_geometry_rows:
        position = int(row["canonical_position"])
        distance = _retained_na_distance(row)
        if distance is None:
            continue
        if distance <= DIRECT_CONTACT_DISTANCE_ANGSTROM:
            direct_contact_positions.add(position)
        elif distance <= NA_FACING_DISTANCE_ANGSTROM:
            annulus_positions.add(position)
    thumb_positions = set(WANG_THUMB_CONTACT_TRACK_POSITIONS)
    annulus_positions = (annulus_positions & mapped) - direct_contact_positions - static_positions - thumb_positions
    distal_positions = mapped - static_positions - thumb_positions - annulus_positions - direct_contact_positions
    return (
        *_STATIC_REGION_SPECS,
        LocalStructureRegionSpec(
            region_id="near_retained_dna_rna_annulus",
            label="Near retained DNA/RNA annulus",
            role="substrate_proximal_review",
            positions=tuple(sorted(annulus_positions)),
        ),
        LocalStructureRegionSpec(
            region_id="distal_scaffold_control",
            label="Distal scaffold control",
            role="distal_scaffold_control",
            positions=tuple(sorted(distal_positions)),
        ),
    )


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
            "status": "insufficient_region_overlap",
            "status_reason": f"shared region C-alpha count {len(shared_positions)} is below {MIN_REGION_CA}",
        }
    displacements = [
        float(np.linalg.norm(candidate_ca[position] - reference_ca[position])) for position in shared_positions
    ]
    squared = sum(displacement * displacement for displacement in displacements) / len(displacements)
    return {
        **base,
        "local_ca_rmsd_angstrom": round(math.sqrt(squared), 3),
        "mean_ca_displacement_angstrom": round(sum(displacements) / len(displacements), 3),
        "max_ca_displacement_angstrom": round(max(displacements), 3),
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
        "coordinate_scope": COORDINATE_SCOPE,
        "n_reference_ca": n_reference_ca,
        "n_candidate_ca": n_candidate_ca,
        "n_shared_ca": n_shared_ca,
        "source_model_path": "" if source_model_path is None else str(source_model_path),
        "reference_model_path": str(reference_model_path),
    }


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


def _retained_na_distance(row: Mapping[str, Any]) -> float | None:
    for field in ("nearest_context_atom_distance_angstrom", "distance_to_retained_na_angstrom"):
        value = row.get(field)
        if value is not None:
            return float(value)
    return None


__all__ = [
    "COORDINATE_SCOPE",
    "LOCAL_STRUCTURE_REGION_IDS",
    "LocalStructureRegionSpec",
    "build_local_structure_region_rows",
    "local_structure_region_specs",
    "mapped_positions_from_residue_map",
]
