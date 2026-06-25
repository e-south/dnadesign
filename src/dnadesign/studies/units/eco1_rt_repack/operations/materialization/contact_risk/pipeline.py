"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact_risk/pipeline.py

Materialize Eco1 RT contact-risk audit evidence for the selected mask.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.paths import DEFAULT_THREAD_OUTPUT_ROOT

_DEFAULT_OUTPUT_ROOT = DEFAULT_THREAD_OUTPUT_ROOT
_CREATED_BY = "dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_risk"
_DEFAULT_CREATED_AT = "2026-06-22T00:00:00Z"
_POLICY_ID = "eco1_rt_contact_risk_audit_v1"


@dataclass(frozen=True)
class MaterializedContactRiskArtifacts:
    """Paths emitted by one Eco1 contact-risk materialization pass."""

    contact_risk_profile_path: Path


def materialize_contact_risk_profile(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    created_at: str = _DEFAULT_CREATED_AT,
) -> MaterializedContactRiskArtifacts:
    """Materialize a non-authoritative contact-risk audit profile."""

    root = (repo_root or _find_repo_root(Path.cwd())).expanduser().resolve()
    out_root = _resolve_path(root, output_root or _DEFAULT_OUTPUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    paths = _required_paths(out_root)
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    residue_rows = pq.read_table(paths["residue_map"]).to_pylist()
    contact_rows = pq.read_table(paths["contact_profile"]).to_pylist()
    contact_geometry_rows = pq.read_table(paths["contact_geometry_profile"]).to_pylist()
    conservation_rows = pq.read_table(paths["conservation_profile"]).to_pylist()
    manual_authority = _load_yaml(paths["manual_mask_authority"])
    mask_set = _load_yaml(paths["mask_set"])

    rows = _contact_risk_rows(
        residue_rows=residue_rows,
        contact_rows=contact_rows,
        contact_geometry_rows=contact_geometry_rows,
        conservation_rows=conservation_rows,
        manual_authority=manual_authority,
        mask_set=mask_set,
    )
    profile = _build_contact_risk_profile(
        rows=rows,
        mask_set=mask_set,
        manual_authority=manual_authority,
        upstream_hashes={name: "sha256:" + _sha256(path) for name, path in paths.items()},
        created_at=created_at,
    )
    output_path = out_root / "contact_risk_profile.yaml"
    output_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    return MaterializedContactRiskArtifacts(contact_risk_profile_path=output_path)


def _required_paths(output_root: Path) -> dict[str, Path]:
    return {
        "residue_map": output_root / "residue_map.parquet",
        "contact_profile": output_root / "contact_profile.parquet",
        "contact_geometry_profile": output_root / "contact_geometry_profile.parquet",
        "conservation_profile": output_root / "conservation_profile.parquet",
        "manual_mask_authority": output_root / "manual_mask_authority.yaml",
        "mask_set": output_root / "mask_set.yaml",
    }


def _contact_risk_rows(
    *,
    residue_rows: list[dict[str, Any]],
    contact_rows: list[dict[str, Any]],
    contact_geometry_rows: list[dict[str, Any]],
    conservation_rows: list[dict[str, Any]],
    manual_authority: Mapping[str, Any],
    mask_set: Mapping[str, Any],
) -> list[dict[str, Any]]:
    contact_by_position = {int(row["canonical_position"]): row for row in contact_rows}
    contact_geometry_by_position = {int(row["canonical_position"]): row for row in contact_geometry_rows}
    conservation_by_position: dict[int, list[dict[str, Any]]] = {}
    for row in conservation_rows:
        conservation_by_position.setdefault(int(row["canonical_position"]), []).append(row)
    manual_by_position = {
        int(row["canonical_position"]): row
        for row in _require_list(manual_authority.get("residues"), "manual_mask_authority.residues")
        if isinstance(row, Mapping)
    }
    candidate_by_position = {
        int(row["canonical_position"]): row
        for row in _require_list(
            manual_authority.get("candidate_prior_residues"),
            "manual_mask_authority.candidate_prior_residues",
        )
        if isinstance(row, Mapping)
    }
    mask_by_position = {
        int(row["canonical_position"]): row
        for row in _require_list(mask_set.get("residues"), "mask_set.residues")
        if isinstance(row, Mapping)
    }

    rows: list[dict[str, Any]] = []
    for residue in residue_rows:
        position = int(residue["canonical_position"])
        contact = contact_by_position[position]
        contact_geometry = contact_geometry_by_position[position]
        profile_rows = conservation_by_position.get(position, [])
        manual = manual_by_position.get(position)
        candidate = candidate_by_position.get(position)
        mask_row = mask_by_position[position]
        selected_conservation_rows = [
            row for row in profile_rows if row.get("profile_id") == "ec86_clade9_conservation_v1"
        ]
        conservation_mask = any(row.get("passes_conservation_mask") is True for row in selected_conservation_rows)
        distance = contact.get("nearest_context_atom_distance_angstrom")
        risk_class, risk_reason = _contact_risk_class(
            mapping_status=str(residue["mapping_status"]),
            manual_mask=manual is not None,
            conservation_mask=conservation_mask,
            wang_candidate_prior=candidate is not None,
            nearest_distance=distance,
        )
        rows.append(
            {
                "canonical_position": position,
                "wt_aa": residue["wt_aa"],
                "structure_chain_id": residue["structure_chain_id"],
                "structure_residue_id": residue["structure_residue_id"],
                "design_position": residue["design_position"],
                "mapping_status": residue["mapping_status"],
                "nearest_context_atom_distance_angstrom": distance,
                "nearest_context_molecule_type": contact.get("nearest_context_molecule_type", ""),
                "nearest_context_chain_id": contact.get("nearest_context_chain_id", ""),
                "nearest_dna_distance_angstrom": contact.get("nearest_dna_distance_angstrom"),
                "nearest_rna_distance_angstrom": contact.get("nearest_rna_distance_angstrom"),
                "sidechain_context_distance_angstrom": contact_geometry.get(
                    "nearest_sidechain_context_distance_angstrom"
                ),
                "backbone_context_distance_angstrom": contact_geometry.get(
                    "nearest_backbone_context_distance_angstrom"
                ),
                "sidechain_atom_status": contact_geometry.get("sidechain_atom_status", ""),
                "contact_atom_count_within_4a": contact_geometry.get("contact_atom_count_within_4a", 0),
                "contact_atom_count_within_6a": contact_geometry.get("contact_atom_count_within_6a", 0),
                "contact_atom_count_within_8a": contact_geometry.get("contact_atom_count_within_8a", 0),
                "contact_atom_count_within_10a": contact_geometry.get("contact_atom_count_within_10a", 0),
                "contact_atom_count_within_12a": contact_geometry.get("contact_atom_count_within_12a", 0),
                "contact_atom_count_within_15a": contact_geometry.get("contact_atom_count_within_15a", 0),
                "contact_atom_count_within_20a": contact_geometry.get("contact_atom_count_within_20a", 0),
                "retained_context_chain_count_within_8a": contact_geometry.get(
                    "retained_context_chain_count_within_8a", 0
                ),
                "retained_context_chain_count_within_12a": contact_geometry.get(
                    "retained_context_chain_count_within_12a", 0
                ),
                "retained_context_chain_count_within_15a": contact_geometry.get(
                    "retained_context_chain_count_within_15a", 0
                ),
                "retained_context_chain_count_within_20a": contact_geometry.get(
                    "retained_context_chain_count_within_20a", 0
                ),
                "contact_threshold_angstrom": contact["contact_threshold_angstrom"],
                "direct_contact_mask": contact["passes_contact_mask"],
                "manual_mask": manual is not None,
                "manual_mask_reason": "" if manual is None else str(manual["manual_mask_reason"]),
                "conservation_mask": conservation_mask,
                "conservation_profile_ids": sorted(
                    row["profile_id"]
                    for row in selected_conservation_rows
                    if row.get("passes_conservation_mask") is True
                ),
                "wang_candidate_prior": candidate is not None,
                "wang_candidate_prior_status": ("" if candidate is None else str(candidate["policy"])),
                "contact_risk_class": risk_class,
                "contact_risk_reason": risk_reason,
                "selected_mask_protected": mask_row["protected"],
                "selected_mask_non_fixed": mask_row["non_fixed"],
                "selected_mask_non_fixed_missing_backbone": mask_row["non_fixed_missing_backbone"],
            }
        )
    return rows


def _contact_risk_class(
    *,
    mapping_status: str,
    manual_mask: bool,
    conservation_mask: bool,
    wang_candidate_prior: bool,
    nearest_distance: Any,
) -> tuple[str, str]:
    if mapping_status != "mapped":
        return (
            "missing_backbone_non_fixed",
            "terminal residues without coordinates are unprotected but not directly mutable",
        )
    if manual_mask:
        return "motif_anchor_protected", "audited NAxxH, YADD, or VTG motif anchor protects this residue"
    if wang_candidate_prior:
        return "wang_interface_candidate_prior", "Wang/Ec86 interface prior protects this residue"
    if conservation_mask:
        return "evolutionarily_conserved", "Eco1 residue passes the clade 9 25% WT plurality conservation rule"
    if nearest_distance is None:
        return "missing_contact_distance", "mapped residue lacks contact distance evidence"
    distance = float(nearest_distance)
    if distance <= 8.0:
        return "high_proximity_retained_context", "nearest retained-context atom is within 8 A"
    if distance <= 12.0:
        return "moderate_proximity_retained_context", "nearest retained-context atom is within 12 A"
    if distance <= 15.0:
        return "diffuse_context_within_15a", "nearest retained-context atom is outside direct contact but inside 15 A"
    if distance <= 20.0:
        return "diffuse_context_within_20a", "nearest retained-context atom is outside direct contact but inside 20 A"
    return "distal_by_nearest_atom", "nearest retained-context atom exceeds 20 A"


def _build_contact_risk_profile(
    *,
    rows: list[dict[str, Any]],
    mask_set: Mapping[str, Any],
    manual_authority: Mapping[str, Any],
    upstream_hashes: Mapping[str, str],
    created_at: str,
) -> dict[str, Any]:
    candidate_positions = [row["canonical_position"] for row in rows if row["wang_candidate_prior"]]
    return {
        "schema_id": "eco1_rt_repack.contact_risk_profile",
        "schema_version": 1,
        "artifact_id": "eco1_rt_conservative_v1.contact_risk_profile",
        "status": "materialized",
        "created_by": _CREATED_BY,
        "created_at": created_at,
        "contact_risk_policy_id": _POLICY_ID,
        "profile_id": mask_set.get("profile_id", "eco1_rt_v1"),
        "sampling_decision": {
            "status": "not_sampling_authoritative",
            "reason": "contact-risk profile audits the selected simple mask and does not create backend requests",
        },
        "evidence_availability": _evidence_availability(),
        "upstream_artifact_hashes": dict(upstream_hashes),
        "source_basis": manual_authority.get("source_basis", []),
        "summary": _summary(rows, mask_set=mask_set, candidate_positions=candidate_positions),
        "residues": rows,
    }


def _evidence_availability() -> dict[str, dict[str, str]]:
    return {
        "nearest_context_atom_distance": {
            "status": "materialized",
            "reason": "retained-context nearest atom distances are present in contact_profile.parquet",
        },
        "sidechain_context_distance": {
            "status": "materialized",
            "reason": "atom-class side-chain distances are present in contact_geometry_profile.parquet",
        },
        "backbone_context_distance": {
            "status": "materialized",
            "reason": "atom-class backbone distances are present in contact_geometry_profile.parquet",
        },
        "contact_atom_density": {
            "status": "materialized",
            "reason": "thresholded retained-context atom counts are present in contact_geometry_profile.parquet",
        },
        "retained_context_chain_count": {
            "status": "materialized",
            "reason": "thresholded retained-context chain counts are present in contact_geometry_profile.parquet",
        },
    }


def _summary(
    rows: list[dict[str, Any]],
    *,
    mask_set: Mapping[str, Any],
    candidate_positions: list[int],
) -> dict[str, Any]:
    return {
        "total_positions": len(rows),
        "mapped_position_count": sum(1 for row in rows if row["mapping_status"] == "mapped"),
        "manual_mask_position_count": sum(1 for row in rows if row["manual_mask"]),
        "wang_candidate_prior_position_count": len(candidate_positions),
        "wang_candidate_prior_positions": candidate_positions,
        "conservation_fixed_position_count": sum(1 for row in rows if row["conservation_mask"]),
        "direct_contact_fixed_position_count": sum(1 for row in rows if row["direct_contact_mask"]),
        "selected_mask_protected_position_count": sum(1 for row in rows if row["selected_mask_protected"]),
        "selected_mask_non_fixed_mapped_position_count": sum(1 for row in rows if row["selected_mask_non_fixed"]),
        "selected_mask_non_fixed_missing_backbone_position_count": sum(
            1 for row in rows if row["selected_mask_non_fixed_missing_backbone"]
        ),
    }


def _resolve_path(repo_root: Path, path: Path) -> Path:
    resolved = path.expanduser()
    return resolved if resolved.is_absolute() else (repo_root / resolved).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return loaded


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _find_repo_root(start: Path) -> Path:
    for parent in (start.resolve(), *start.resolve().parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("repo root with pyproject.toml not found")
