"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/colabfold/outputs.py

ColabFold output normalization into generic fold-check rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dnadesign.thread.adapters.colabfold.index import ColabFoldOutputIndex
from dnadesign.thread.adapters.colabfold.manifest import (
    file_sha256_uri,
    ordered_positions_hash,
    runtime_parameters_hash,
)
from dnadesign.thread.adapters.colabfold.metrics import ca_coordinates, ca_rmsd, mean_ca_plddt, pae_summary

MAPPED_REFERENCE_COORDINATE_BASIS = "reference_ca_order_to_one_based_mobile_sequence_position"
EQUAL_LENGTH_REFERENCE_COORDINATE_BASIS = "equal_length_reference_ca_order_to_mobile_ca_order"


def build_colabfold_foldcheck_rows(
    *,
    output_root: Path,
    request_manifest: Mapping[str, Any],
    runtime_version: str,
    runtime_parameters: Mapping[str, Any],
    reference_pdb_path: Path | None = None,
    reference_mobile_positions: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Build generic fold-check rows from one ColabFold output directory.

    ``reference_mobile_positions`` is an ordered, one-based correspondence from
    each C-alpha in an explicit reference PDB to the matching C-alpha position
    in every mobile ColabFold model. Without it, the reference and mobile
    coordinate arrays retain the original equal-length requirement.
    """

    sequence_rows = _sequence_rows(request_manifest)
    sequence_ids = tuple(str(sequence["sequence_id"]) for sequence in sequence_rows)
    wt_sequence_id = str(request_manifest.get("wt_sequence_id", "wild_type"))
    runtime_hash = runtime_parameters_hash(runtime_parameters)
    output_index = ColabFoldOutputIndex.from_output_root(output_root, sequence_ids=sequence_ids)
    has_explicit_reference = reference_pdb_path is not None
    wt_baseline_coords = _wt_baseline_coordinates(output_index=output_index, wt_sequence_id=wt_sequence_id)
    reference_coords = _reference_coordinates(
        wt_baseline_coords=wt_baseline_coords,
        reference_pdb_path=reference_pdb_path,
    )
    mobile_positions = _validated_reference_mobile_positions(
        reference_mobile_positions,
        reference_coords=reference_coords,
        reference_pdb_path=reference_pdb_path,
    )
    reference_lineage = _reference_lineage(
        reference_pdb_path=reference_pdb_path,
        reference_mobile_positions=mobile_positions,
        reference_ca_count=0 if reference_coords is None else len(reference_coords),
    )

    rows: list[dict[str, Any]] = []
    for sequence in sequence_rows:
        candidate_id = str(sequence["sequence_id"])
        model_path = output_index.select_model_pdb(candidate_id)
        score_path = output_index.select_score_json(candidate_id)
        if model_path is None:
            rows.append(
                _failure_row(
                    candidate_id=candidate_id,
                    sequence_hash=str(sequence.get("sequence_hash", "")),
                    request_manifest=request_manifest,
                    runtime_version=runtime_version,
                    runtime_parameters_hash=runtime_hash,
                    wt_sequence_id=wt_sequence_id,
                    reason="colabfold_output_missing",
                    reference_lineage=reference_lineage,
                )
            )
            continue
        mobile_coords = ca_coordinates(model_path)
        declared_length = _declared_sequence_length(sequence, candidate_id=candidate_id)
        if len(mobile_coords) != declared_length:
            rows.append(
                _failure_row(
                    candidate_id=candidate_id,
                    sequence_hash=str(sequence.get("sequence_hash", "")),
                    request_manifest=request_manifest,
                    runtime_version=runtime_version,
                    runtime_parameters_hash=runtime_hash,
                    wt_sequence_id=wt_sequence_id,
                    reason="colabfold_ca_count_mismatch",
                    reference_lineage=reference_lineage,
                )
            )
            continue
        plddt = mean_ca_plddt(model_path)
        reference_rmsd = (
            0.0
            if candidate_id == wt_sequence_id and reference_pdb_path is None
            else _rmsd_to_reference(
                mobile_coords,
                reference_coords=reference_coords,
                reference_mobile_positions=mobile_positions,
                candidate_id=candidate_id,
            )
        )
        wt_baseline_rmsd = (
            0.0
            if candidate_id == wt_sequence_id and wt_baseline_coords is not None
            else (ca_rmsd(mobile_coords, wt_baseline_coords) if wt_baseline_coords is not None else None)
        )
        if plddt is None or reference_rmsd is None or wt_baseline_rmsd is None:
            reason = "colabfold_required_metric_missing"
            rows.append(
                _failure_row(
                    candidate_id=candidate_id,
                    sequence_hash=str(sequence.get("sequence_hash", "")),
                    request_manifest=request_manifest,
                    runtime_version=runtime_version,
                    runtime_parameters_hash=runtime_hash,
                    wt_sequence_id=wt_sequence_id,
                    reason=reason,
                    reference_lineage=reference_lineage,
                )
            )
            continue
        row = {
            "candidate_id": candidate_id,
            "runtime_kind": str(request_manifest.get("runtime_kind", "alphafold_family_colabfold")),
            "runtime_version": runtime_version,
            "input_sequence_hash": str(sequence.get("sequence_hash", "")),
            "reference_structure_id": str(request_manifest.get("reference_structure_id", "")),
            "wt_baseline_artifact_id": "self" if candidate_id == wt_sequence_id else wt_sequence_id,
            "runtime_parameters_hash": runtime_hash,
            "threshold_id": str(request_manifest.get("threshold_policy_id", "")),
            "threshold_values": dict(_mapping_or_empty(request_manifest.get("threshold_values"))),
            "plddt": plddt,
            "pae_summary": pae_summary(score_path),
            "backbone_rmsd_to_reference": reference_rmsd,
            "protected_contact_retention": None,
            "status": "accepted",
            "rejection_reason": "",
            "missing_metric_reason": "",
            "model_artifact_path": str(model_path),
            "score_artifact_path": str(score_path) if score_path is not None else "",
        }
        if has_explicit_reference:
            row["backbone_rmsd_to_wt_baseline"] = wt_baseline_rmsd
            row.update(reference_lineage)
        rows.append(row)
    return rows


def _sequence_rows(request_manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    sequences = request_manifest.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("ColabFold normalization requires request manifest sequences")
    rows = [row for row in sequences if isinstance(row, Mapping)]
    if len(rows) != len(sequences):
        raise ValueError("ColabFold normalization request sequences must be mappings")
    return rows


def _declared_sequence_length(sequence: Mapping[str, Any], *, candidate_id: str) -> int:
    value = sequence.get("length")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"ColabFold request sequence {candidate_id!r} must declare a positive integer length")
    return value


def _reference_coordinates(
    *,
    wt_baseline_coords: Any | None,
    reference_pdb_path: Path | None,
) -> Any | None:
    if reference_pdb_path is not None:
        if not reference_pdb_path.exists():
            raise FileNotFoundError(reference_pdb_path)
        return ca_coordinates(reference_pdb_path)
    return wt_baseline_coords


def _wt_baseline_coordinates(*, output_index: ColabFoldOutputIndex, wt_sequence_id: str) -> Any | None:
    wt_model_path = output_index.select_model_pdb(wt_sequence_id)
    if wt_model_path is None:
        return None
    return ca_coordinates(wt_model_path)


def _validated_reference_mobile_positions(
    value: Sequence[int] | None,
    *,
    reference_coords: Any | None,
    reference_pdb_path: Path | None,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if reference_pdb_path is None:
        raise ValueError("reference_mobile_positions requires an explicit reference_pdb_path")
    reference_count = 0 if reference_coords is None else len(reference_coords)
    positions = tuple(value)
    if not positions:
        raise ValueError("reference_mobile_positions must contain at least one position")
    if any(not isinstance(position, int) or isinstance(position, bool) or position < 1 for position in positions):
        raise ValueError("reference_mobile_positions must contain positive one-based integers")
    if len(set(positions)) != len(positions):
        raise ValueError("reference_mobile_positions must not contain duplicate positions")
    if len(positions) != reference_count:
        raise ValueError(
            "reference mobile position count must match explicit reference C-alpha count; "
            f"observed {len(positions)} positions and {reference_count} reference coordinates at {reference_pdb_path}"
        )
    return positions


def _reference_lineage(
    *,
    reference_pdb_path: Path | None,
    reference_mobile_positions: tuple[int, ...] | None,
    reference_ca_count: int,
) -> dict[str, str]:
    if reference_pdb_path is None:
        return {}
    positions = (
        tuple(range(1, reference_ca_count + 1)) if reference_mobile_positions is None else reference_mobile_positions
    )
    return {
        "reference_structure_hash": file_sha256_uri(reference_pdb_path),
        "reference_mobile_positions_hash": ordered_positions_hash(positions),
        "reference_coordinate_basis": (
            EQUAL_LENGTH_REFERENCE_COORDINATE_BASIS
            if reference_mobile_positions is None
            else MAPPED_REFERENCE_COORDINATE_BASIS
        ),
    }


def _rmsd_to_reference(
    mobile_coords: Any,
    *,
    reference_coords: Any | None,
    reference_mobile_positions: tuple[int, ...] | None,
    candidate_id: str,
) -> float | None:
    if reference_coords is None:
        return None
    selected_mobile = mobile_coords
    if reference_mobile_positions is not None:
        maximum_position = max(reference_mobile_positions)
        if maximum_position > len(mobile_coords):
            raise ValueError(
                f"reference mobile position {maximum_position} is outside candidate {candidate_id!r} "
                f"C-alpha coordinate range 1..{len(mobile_coords)}"
            )
        selected_mobile = mobile_coords[[position - 1 for position in reference_mobile_positions]]
    return ca_rmsd(selected_mobile, reference_coords)


def _failure_row(
    *,
    candidate_id: str,
    sequence_hash: str,
    request_manifest: Mapping[str, Any],
    runtime_version: str,
    runtime_parameters_hash: str,
    wt_sequence_id: str,
    reason: str,
    reference_lineage: Mapping[str, str],
) -> dict[str, Any]:
    row = {
        "candidate_id": candidate_id,
        "runtime_kind": str(request_manifest.get("runtime_kind", "alphafold_family_colabfold")),
        "runtime_version": runtime_version,
        "input_sequence_hash": sequence_hash,
        "reference_structure_id": str(request_manifest.get("reference_structure_id", "")),
        "wt_baseline_artifact_id": "self" if candidate_id == wt_sequence_id else wt_sequence_id,
        "runtime_parameters_hash": runtime_parameters_hash,
        "threshold_id": str(request_manifest.get("threshold_policy_id", "")),
        "threshold_values": dict(_mapping_or_empty(request_manifest.get("threshold_values"))),
        "plddt": None,
        "pae_summary": {"status": "not_available"},
        "backbone_rmsd_to_reference": None,
        "protected_contact_retention": None,
        "status": "errored",
        "rejection_reason": reason,
        "missing_metric_reason": reason,
        "model_artifact_path": "",
        "score_artifact_path": "",
    }
    if reference_lineage:
        row["backbone_rmsd_to_wt_baseline"] = None
        row.update(reference_lineage)
    return row


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
