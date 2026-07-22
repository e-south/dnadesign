"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/structure/preprocessing.py

Structure-preprocessing manifest validators for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.structure.provenance import (
    load_yaml_mapping,
    resolve_contract_ref,
    validate_upstream_artifact_hashes,
)

_DOCS_ROOT = Path("docs/studies/eco1_rt_repack")
_STRUCTURE_PREPROCESSING = _DOCS_ROOT / "workbench/provenance/structure-preprocessing.yaml"
_STRUCTURE_SOURCES = _DOCS_ROOT / "workbench/provenance/structure-sources.yaml"
_NUMBERING_POLICY = _DOCS_ROOT / "workbench/provenance/residue-numbering-policy.yaml"


def validate_structure_preprocessing_manifest_content(
    path: Path,
    *,
    repo_root: Path,
    structure_root: Path,
) -> list[ContractIssue]:
    """Validate dimer-to-protomer preprocessing provenance and upstream hashes."""

    issues: list[ContractIssue] = []
    manifest = load_yaml_mapping(path)
    preprocessing = load_yaml_mapping(repo_root / _STRUCTURE_PREPROCESSING)
    structure_sources = load_yaml_mapping(repo_root / _STRUCTURE_SOURCES)
    selected_source = _require_mapping(structure_sources.get("selected_source"), "selected_source")
    backbone_bundle = load_yaml_mapping(structure_root / "backbone_bundle.yaml")

    expected_pairs = {
        "schema_id": "eco1_rt_repack.structure_preprocessing_manifest",
        "status": "materialized",
        "preprocessing_id": preprocessing.get("preprocessing_id"),
    }
    for field in ("schema_version", "artifact_id", "created_by", "created_at", "upstream_artifact_hashes"):
        if not manifest.get(field):
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.structure.structure_preprocessing_missing_lifecycle_field",
                    message=f"structure_preprocessing_manifest.yaml must declare lifecycle field {field!r}",
                    path=f"{path}:{field}",
                )
            )
    for field, expected in expected_pairs.items():
        if manifest.get(field) != expected:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.structure.structure_preprocessing_field_mismatch",
                    message=f"structure_preprocessing_manifest.yaml field {field!r} must equal {expected!r}",
                    path=f"{path}:{field}",
                )
            )

    selected = manifest.get("selected_protomer")
    if not isinstance(selected, Mapping):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.structure_preprocessing_missing_selected_protomer",
                message="structure_preprocessing_manifest.yaml must declare selected_protomer",
                path=f"{path}:selected_protomer",
            )
        )
    else:
        _validate_selected_protomer(issues, selected=selected, selected_source=selected_source, path=path)
    _validate_design_objectives(
        issues,
        manifest=manifest,
        preprocessing=preprocessing,
        path=path,
    )
    _validate_chain_inventory(issues, manifest=manifest, backbone_bundle=backbone_bundle, path=path)
    issues.extend(
        validate_upstream_artifact_hashes(
            _require_mapping_or_none(manifest.get("upstream_artifact_hashes")),
            structure_preprocessing_upstream_artifact_paths(repo_root, structure_root),
            path=path,
            check_id="eco1_rt.structure.structure_preprocessing_upstream_hash_mismatch",
            artifact_label="structure_preprocessing_manifest.yaml",
        )
    )
    return issues


def structure_preprocessing_upstream_artifact_paths(repo_root: Path, structure_root: Path) -> dict[str, Path]:
    """Return current upstream paths required by structure_preprocessing_manifest.yaml."""

    structure_sources = load_yaml_mapping(repo_root / _STRUCTURE_SOURCES)
    selected_source = _require_mapping(structure_sources.get("selected_source"), "selected_source")
    return {
        "structure_preprocessing_yaml": repo_root / _STRUCTURE_PREPROCESSING,
        "structure_sources_yaml": repo_root / _STRUCTURE_SOURCES,
        "residue_numbering_policy_yaml": repo_root / _NUMBERING_POLICY,
        "backbone_bundle": structure_root / "backbone_bundle.yaml",
        "ec86kit_manifest": resolve_contract_ref(repo_root, _require_text(selected_source, "ec86kit_manifest_ref")),
        "ec86kit_model": resolve_contract_ref(repo_root, _require_text(selected_source, "ec86kit_model_ref")),
    }


def contact_geometry_upstream_artifact_paths(repo_root: Path, structure_root: Path) -> dict[str, Path]:
    """Return current upstream paths required by contact_geometry_profile.parquet."""

    structure_sources = load_yaml_mapping(repo_root / _STRUCTURE_SOURCES)
    selected_source = _require_mapping(structure_sources.get("selected_source"), "selected_source")
    return {
        "structure_sources_yaml": repo_root / _STRUCTURE_SOURCES,
        "structure_preprocessing_manifest": structure_root / "structure_preprocessing_manifest.yaml",
        "backbone_bundle": structure_root / "backbone_bundle.yaml",
        "residue_map": structure_root / "residue_map.parquet",
        "ec86kit_model": resolve_contract_ref(repo_root, _require_text(selected_source, "ec86kit_model_ref")),
    }


def _validate_selected_protomer(
    issues: list[ContractIssue],
    *,
    selected: Mapping[str, Any],
    selected_source: Mapping[str, Any],
    path: Path,
) -> None:
    expected_fields = {
        "source_id": selected_source.get("source_id"),
        "rt_chain_id": selected_source.get("rt_chain_id"),
        "retained_context_policy": selected_source.get("retained_context_policy"),
    }
    for field, expected in expected_fields.items():
        if selected.get(field) != expected:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.structure.structure_preprocessing_selected_protomer_mismatch",
                    message=f"selected_protomer.{field} must match selected structure source",
                    path=f"{path}:selected_protomer.{field}",
                )
            )
    if _as_string_list(selected.get("retained_context_chains")) != _as_string_list(
        selected_source.get("retained_context_chains")
    ):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.structure_preprocessing_selected_protomer_mismatch",
                message="selected_protomer.retained_context_chains must match selected structure source",
                path=f"{path}:selected_protomer.retained_context_chains",
            )
        )


def _validate_chain_inventory(
    issues: list[ContractIssue],
    *,
    manifest: Mapping[str, Any],
    backbone_bundle: Mapping[str, Any],
    path: Path,
) -> None:
    manifest_roles = _chain_rows_by_id(manifest.get("chain_inventory"))
    backbone_roles = _chain_rows_by_id(backbone_bundle.get("chain_inventory"))
    for chain_id in ("A", "D", "E", "F"):
        manifest_role = manifest_roles.get(chain_id)
        backbone_role = backbone_roles.get(chain_id)
        if not isinstance(manifest_role, Mapping) or not isinstance(backbone_role, Mapping):
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.structure.structure_preprocessing_chain_inventory_mismatch",
                    message=f"chain_inventory must include selected chain {chain_id!r}",
                    path=f"{path}:chain_inventory",
                )
            )
            continue
        for field in ("molecule_type", "thread_role", "retention"):
            if manifest_role.get(field) != backbone_role.get(field):
                issues.append(
                    ContractIssue(
                        check_id="eco1_rt.structure.structure_preprocessing_chain_inventory_mismatch",
                        message=f"chain_inventory for {chain_id!r} must match backbone_bundle.yaml field {field!r}",
                        path=f"{path}:chain_inventory.{chain_id}",
                    )
                )


def _validate_design_objectives(
    issues: list[ContractIssue],
    *,
    manifest: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    path: Path,
) -> None:
    expected = preprocessing.get("design_objectives")
    observed = manifest.get("design_objectives")
    if observed != expected:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.structure_preprocessing_design_objective_mismatch",
                message=(
                    "structure_preprocessing_manifest.yaml design_objectives must match structure-preprocessing.yaml"
                ),
                path=f"{path}:design_objectives",
            )
        )
        return
    if not isinstance(observed, Mapping):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.structure_preprocessing_missing_design_objectives",
                message="structure_preprocessing_manifest.yaml must declare design_objectives",
                path=f"{path}:design_objectives",
            )
        )
        return
    if observed.get("preserve_paired_protomer_dimerization") is not False:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.structure_preprocessing_dimer_objective_not_explicit",
                message="Eco1 mask evidence must explicitly record paired-protomer dimerization as non-objective",
                path=f"{path}:design_objectives.preserve_paired_protomer_dimerization",
            )
        )


def _chain_rows_by_id(value: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list):
        return {}
    return {
        str(row["chain_id"]): row
        for row in value
        if isinstance(row, Mapping) and isinstance(row.get("chain_id"), str) and row["chain_id"].strip()
    }


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _require_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()
