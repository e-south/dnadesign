"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/colabfold/outputs.py

ColabFold output normalization into generic fold-check rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dnadesign.thread.adapters.colabfold.manifest import runtime_parameters_hash
from dnadesign.thread.adapters.colabfold.metrics import ca_coordinates, ca_rmsd, mean_ca_plddt, pae_summary


def build_colabfold_foldcheck_rows(
    *,
    output_root: Path,
    request_manifest: Mapping[str, Any],
    runtime_version: str,
    runtime_parameters: Mapping[str, Any],
    reference_pdb_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Build generic fold-check rows from one ColabFold output directory."""

    sequence_rows = _sequence_rows(request_manifest)
    wt_sequence_id = str(request_manifest.get("wt_sequence_id", "wild_type"))
    runtime_hash = runtime_parameters_hash(runtime_parameters)
    reference_coords = _reference_coordinates(
        output_root=output_root,
        wt_sequence_id=wt_sequence_id,
        reference_pdb_path=reference_pdb_path,
    )

    rows: list[dict[str, Any]] = []
    for sequence in sequence_rows:
        candidate_id = str(sequence["sequence_id"])
        model_path = _select_model_pdb(output_root, candidate_id)
        score_path = _select_score_json(output_root, candidate_id)
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
                )
            )
            continue
        plddt = mean_ca_plddt(model_path)
        mobile_coords = ca_coordinates(model_path)
        rmsd = (
            0.0
            if candidate_id == wt_sequence_id
            else (ca_rmsd(mobile_coords, reference_coords) if reference_coords is not None else None)
        )
        if plddt is None or rmsd is None:
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
                )
            )
            continue
        rows.append(
            {
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
                "backbone_rmsd_to_reference": rmsd,
                "protected_contact_retention": None,
                "status": "accepted",
                "rejection_reason": "",
                "missing_metric_reason": "",
                "model_artifact_path": str(model_path),
                "score_artifact_path": str(score_path) if score_path is not None else "",
            }
        )
    return rows


def _sequence_rows(request_manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    sequences = request_manifest.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("ColabFold normalization requires request manifest sequences")
    rows = [row for row in sequences if isinstance(row, Mapping)]
    if len(rows) != len(sequences):
        raise ValueError("ColabFold normalization request sequences must be mappings")
    return rows


def _reference_coordinates(
    *,
    output_root: Path,
    wt_sequence_id: str,
    reference_pdb_path: Path | None,
) -> Any | None:
    if reference_pdb_path is not None:
        if not reference_pdb_path.exists():
            raise FileNotFoundError(reference_pdb_path)
        return ca_coordinates(reference_pdb_path)
    wt_model_path = _select_model_pdb(output_root, wt_sequence_id)
    if wt_model_path is None:
        return None
    return ca_coordinates(wt_model_path)


def _select_model_pdb(output_root: Path, sequence_id: str) -> Path | None:
    candidates = [
        path
        for path in output_root.rglob("*.pdb")
        if path.name == f"{sequence_id}.pdb" or path.name.startswith(f"{sequence_id}_")
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (_rank_key(path.name), "relaxed" not in path.name, path.name))[0]


def _select_score_json(output_root: Path, sequence_id: str) -> Path | None:
    candidates = [
        path
        for path in output_root.rglob("*.json")
        if (path.name == f"{sequence_id}.json" or path.name.startswith(f"{sequence_id}_"))
        and ("score" in path.name or "pae" in path.name or "aligned_error" in path.name)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (_rank_key(path.name), path.name))[0]


def _rank_key(name: str) -> int:
    for token in name.replace(".", "_").split("_"):
        if token.isdigit():
            return int(token)
        if token.startswith("rank") and token.removeprefix("rank").isdigit():
            return int(token.removeprefix("rank"))
    return 9999


def _failure_row(
    *,
    candidate_id: str,
    sequence_hash: str,
    request_manifest: Mapping[str, Any],
    runtime_version: str,
    runtime_parameters_hash: str,
    wt_sequence_id: str,
    reason: str,
) -> dict[str, Any]:
    return {
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


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
