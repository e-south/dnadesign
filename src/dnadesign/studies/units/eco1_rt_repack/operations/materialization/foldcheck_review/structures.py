"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_review/structures.py

Structure-panel file staging for Eco1 fold-check review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review.constants import (
    FULL_STRUCTURE_SET_SCHEMA_ID,
    STRUCTURE_PANEL_SCHEMA_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review.models import PanelEntry
from dnadesign.thread.adapters.proteinmpnn.hashing import sha256_uri


def stage_structure_panel(
    *,
    structures_root: Path,
    reference_backbone_path: Path,
    wt_fold_row: Mapping[str, Any],
    selected_rows: list[Mapping[str, Any]],
    structure_panel_path: Path,
    source_request_hash: str,
    fallback_model_root: Path | None = None,
) -> list[PanelEntry]:
    """Copy selected local structures and write the structure-panel manifest."""

    structures_root.mkdir(parents=True, exist_ok=True)
    reference_entry = _stage_reference(reference_backbone_path, structures_root)
    entries = [
        _stage_model_entry(
            row=wt_fold_row,
            selection_stratum="wild_type_runtime_baseline",
            structures_root=structures_root,
            fallback_model_root=fallback_model_root,
        )
    ]
    entries.extend(
        _stage_model_entry(
            row=row,
            selection_stratum=str(row["selection_stratum"]),
            structures_root=structures_root,
            fallback_model_root=fallback_model_root,
        )
        for row in selected_rows
    )
    manifest = {
        "schema_id": STRUCTURE_PANEL_SCHEMA_ID,
        "schema_version": 1,
        "status": "materialized",
        "path_policy": "local_paths_manifest_relative",
        "source_request_hash": source_request_hash,
        "rmsd_semantics": {
            "wt_runtime_ca_rmsd": "C-alpha RMSD to the ColabFold WT runtime model from foldcheck_report.parquet",
            "cryoem_mapped_ca_rmsd": (
                "C-alpha RMSD over mapped Eco1 positions to the ec86kit/7V9U protein backbone when model PDBs are local"
            ),
        },
        "reference_structure": _relative_reference_entry(reference_entry, manifest_root=structure_panel_path.parent),
        "selected_structures": [
            _relative_panel_entry(entry, manifest_root=structure_panel_path.parent) for entry in entries
        ],
    }
    structure_panel_path.parent.mkdir(parents=True, exist_ok=True)
    structure_panel_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return entries


def stage_full_structure_set(
    *,
    structures_root: Path,
    reference_backbone_path: Path,
    wt_fold_row: Mapping[str, Any],
    ranking_rows: list[Mapping[str, Any]],
    full_structure_set_path: Path,
    source_request_hash: str,
) -> list[PanelEntry]:
    """Stage one local PDB per accepted fold-check row and write a complete manifest."""

    structures_root.mkdir(parents=True, exist_ok=True)
    verified_existing_entries = _verified_existing_entries(full_structure_set_path)
    entries = [
        _stage_model_entry(
            row=wt_fold_row,
            selection_stratum="full_fold_set",
            structures_root=structures_root,
            fallback_model_root=None,
            verified_existing_entries=verified_existing_entries,
        )
    ]
    entries.extend(
        _stage_model_entry(
            row=row,
            selection_stratum="full_fold_set",
            structures_root=structures_root,
            fallback_model_root=None,
            verified_existing_entries=verified_existing_entries,
        )
        for row in ranking_rows
    )
    copy_summary: dict[str, int] = {}
    for entry in entries:
        copy_summary[entry.copy_status] = copy_summary.get(entry.copy_status, 0) + 1
    manifest = {
        "schema_id": FULL_STRUCTURE_SET_SCHEMA_ID,
        "schema_version": 1,
        "status": "materialized",
        "path_policy": "local_paths_manifest_relative",
        "source_request_hash": source_request_hash,
        "structure_count": len(entries),
        "copy_summary": copy_summary,
        "structure_policy": {
            "scope": "WT baseline plus all accepted ProteinMPNN candidates from foldcheck_report.parquet",
            "local_storage": "one normalized PDB per fold-check row; raw ColabFold output trees stay on SCC",
            "cryoem_reference_source_path": str(reference_backbone_path),
        },
        "structures": [_relative_panel_entry(entry, manifest_root=full_structure_set_path.parent) for entry in entries],
    }
    full_structure_set_path.parent.mkdir(parents=True, exist_ok=True)
    full_structure_set_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return entries


def _stage_reference(reference_backbone_path: Path, structures_root: Path) -> dict[str, str]:
    local_path = structures_root / "ec86kit_chain_a_backbone_reference.pdb"
    if reference_backbone_path.exists():
        shutil.copyfile(reference_backbone_path, local_path)
        return {
            "structure_id": "ec86kit_7v9u_protomer1_chain_a_backbone",
            "source_path": str(reference_backbone_path),
            "local_path": str(local_path),
            "copy_status": "copied",
            "source_hash": sha256_uri(reference_backbone_path),
        }
    return {
        "structure_id": "ec86kit_7v9u_protomer1_chain_a_backbone",
        "source_path": str(reference_backbone_path),
        "local_path": str(local_path),
        "copy_status": "source_not_local",
        "source_hash": "",
    }


def _stage_model_entry(
    *,
    row: Mapping[str, Any],
    selection_stratum: str,
    structures_root: Path,
    fallback_model_root: Path | None,
    verified_existing_entries: Mapping[str, Mapping[str, Any]] | None = None,
) -> PanelEntry:
    candidate_id = str(row["candidate_id"])
    source_path = _optional_model_artifact_path(row.get("model_artifact_path"))
    local_path = structures_root / f"{candidate_id}.pdb"
    if source_path is not None and source_path.exists():
        shutil.copyfile(source_path, local_path)
        copy_status = "copied"
        source_hash = sha256_uri(source_path)
    elif source_path is not None and (structures_root / source_path.name).exists():
        mirrored_source_path = structures_root / source_path.name
        source_hash = sha256_uri(mirrored_source_path)
        if mirrored_source_path != local_path:
            shutil.copyfile(mirrored_source_path, local_path)
            mirrored_source_path.unlink()
        copy_status = "copied_from_local_source_mirror"
    elif fallback_model_root is not None and (fallback_model_root / f"{candidate_id}.pdb").exists():
        fallback_path = fallback_model_root / f"{candidate_id}.pdb"
        shutil.copyfile(fallback_path, local_path)
        copy_status = "copied_from_local_full_set"
        source_hash = sha256_uri(fallback_path)
    elif local_path.exists():
        source_hash = _verified_existing_source_hash(
            candidate_id=candidate_id,
            source_path=source_path or Path(""),
            local_path=local_path,
            structures_root=structures_root,
            verified_existing_entries=verified_existing_entries,
        )
        copy_status = "already_local_verified"
    else:
        copy_status = "source_not_local"
        source_hash = ""
    return PanelEntry(
        candidate_id=candidate_id,
        selection_stratum=selection_stratum,
        source_model_artifact_path="" if source_path is None else str(source_path),
        local_model_artifact_path=str(local_path),
        copy_status=copy_status,
        source_model_artifact_hash=source_hash,
        display_label=_display_label(row, candidate_id=candidate_id),
        sequence_identity_percent=_sequence_identity_percent(row, candidate_id=candidate_id),
        proteinmpnn_rank=_optional_int(row.get("proteinmpnn_rank")),
        wt_runtime_ca_rmsd=_optional_float(row.get("wt_runtime_ca_rmsd")),
    )


def _optional_model_artifact_path(value: Any) -> Path | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    return Path(text)


def _relative_reference_entry(entry: dict[str, str], *, manifest_root: Path) -> dict[str, str]:
    normalized = dict(entry)
    normalized["local_path"] = _manifest_relative_path(Path(normalized["local_path"]), manifest_root=manifest_root)
    return normalized


def _relative_panel_entry(entry: PanelEntry, *, manifest_root: Path) -> dict[str, Any]:
    normalized = dict(entry.__dict__)
    normalized["local_model_artifact_path"] = _manifest_relative_path(
        Path(entry.local_model_artifact_path),
        manifest_root=manifest_root,
    )
    return normalized


def _verified_existing_entries(manifest_path: Path) -> dict[str, Mapping[str, Any]]:
    if not manifest_path.exists():
        return {}
    loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return {}
    entries = loaded.get("structures")
    if not isinstance(entries, list):
        return {}
    return {
        str(entry["candidate_id"]): entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("candidate_id") is not None
    }


def _verified_existing_source_hash(
    *,
    candidate_id: str,
    source_path: Path,
    local_path: Path,
    structures_root: Path,
    verified_existing_entries: Mapping[str, Mapping[str, Any]] | None,
) -> str:
    existing_entry = (verified_existing_entries or {}).get(candidate_id)
    local_hash = sha256_uri(local_path)
    if existing_entry is None:
        raise ValueError(
            f"Found unverified staged model for {candidate_id!r} at {local_path}; "
            "remove it or regenerate from a reachable source path."
        )
    if str(existing_entry.get("source_model_artifact_path") or "") != str(source_path):
        raise ValueError(
            f"Found unverified staged model for {candidate_id!r}: source path changed from "
            f"{existing_entry.get('source_model_artifact_path')!r} to {str(source_path)!r}."
        )
    recorded_hash = str(existing_entry.get("source_model_artifact_hash") or "")
    if recorded_hash != local_hash:
        raise ValueError(
            f"Found unverified staged model for {candidate_id!r}: local hash does not match the previous manifest."
        )
    previous_local_path = _resolve_manifest_relative_path(
        str(existing_entry.get("local_model_artifact_path") or ""),
        manifest_root=structures_root.parent.parent,
    )
    if previous_local_path != local_path:
        raise ValueError(
            f"Found unverified staged model for {candidate_id!r}: local path does not match the previous manifest."
        )
    return local_hash


def _resolve_manifest_relative_path(value: str, *, manifest_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_root / path


def _display_label(row: Mapping[str, Any], *, candidate_id: str) -> str:
    if candidate_id == "wild_type":
        return "WT ColabFold baseline"
    rank = _optional_int(row.get("proteinmpnn_rank"))
    if rank is not None:
        return f"ProteinMPNN variant rank {rank}"
    short_id = candidate_id.removeprefix("thread_candidate_")
    return f"ProteinMPNN variant {short_id}"


def _sequence_identity_percent(row: Mapping[str, Any], *, candidate_id: str) -> float | None:
    if candidate_id == "wild_type":
        return 100.0
    seq_recovery = row.get("seq_recovery")
    return None if seq_recovery is None else float(seq_recovery) * 100.0


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _manifest_relative_path(path: Path, *, manifest_root: Path) -> str:
    if not path.is_absolute():
        return str(path)
    return os.path.relpath(path, start=manifest_root)
