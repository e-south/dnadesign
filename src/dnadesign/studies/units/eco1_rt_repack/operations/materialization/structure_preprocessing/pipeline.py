"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/structure_preprocessing/pipeline.py

Materialize Eco1 RT structure-preprocessing provenance.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_DOCS_ROOT = Path("docs/studies/eco1_rt_repack")
_STRUCTURE_PREPROCESSING = _DOCS_ROOT / "workbench/provenance/structure-preprocessing.yaml"
_STRUCTURE_SOURCES = _DOCS_ROOT / "workbench/provenance/structure-sources.yaml"
_NUMBERING_POLICY = _DOCS_ROOT / "workbench/provenance/residue-numbering-policy.yaml"
_DEFAULT_OUTPUT_ROOT = Path("outputs/thread/eco1_rt_conservative_v1")
_CREATED_BY = "dnadesign.studies.units.eco1_rt_repack.operations.materialization.structure_preprocessing"
_DEFAULT_CREATED_AT = "2026-06-22T00:00:00Z"


@dataclass(frozen=True)
class MaterializedStructurePreprocessingArtifacts:
    """Paths emitted by one Eco1 structure-preprocessing materialization pass."""

    structure_preprocessing_manifest_path: Path


def materialize_structure_preprocessing_manifest(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    created_at: str = _DEFAULT_CREATED_AT,
) -> MaterializedStructurePreprocessingArtifacts:
    """Materialize selected raw-dimer to protomer provenance as a runtime manifest."""

    root = (repo_root or _find_repo_root(Path.cwd())).expanduser().resolve()
    out_root = _resolve_output_root(root, output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    preprocessing = _load_yaml(root / _STRUCTURE_PREPROCESSING)
    structure_sources = _load_yaml(root / _STRUCTURE_SOURCES)
    numbering_policy = _load_yaml(root / _NUMBERING_POLICY)
    backbone_bundle_path = out_root / "backbone_bundle.yaml"
    if not backbone_bundle_path.exists():
        raise FileNotFoundError(backbone_bundle_path)
    backbone_bundle = _load_yaml(backbone_bundle_path)

    selected_source = _require_mapping(structure_sources.get("selected_source"), "selected_source")
    selected_protomer = _require_mapping(preprocessing.get("selected_protomer"), "selected_protomer")
    _validate_selected_protomer(selected_source=selected_source, selected_protomer=selected_protomer)
    _validate_chain_inventory(preprocessing=preprocessing, backbone_bundle=backbone_bundle)

    manifest_path = _resolve_source_ref(root, _require_text(selected_source, "ec86kit_manifest_ref"))
    model_path = _resolve_source_ref(root, _require_text(selected_source, "ec86kit_model_ref"))
    _require_hash(manifest_path, _require_text(selected_source, "ec86kit_manifest_sha256"))
    _require_hash(model_path, _require_text(selected_source, "ec86kit_model_sha256"))

    manifest = {
        "schema_id": "eco1_rt_repack.structure_preprocessing_manifest",
        "schema_version": 1,
        "artifact_id": "eco1_rt_conservative_v1.structure_preprocessing_manifest",
        "status": "materialized",
        "created_by": _CREATED_BY,
        "created_at": created_at,
        "preprocessing_id": _require_text(preprocessing, "preprocessing_id"),
        "raw_structure": preprocessing["raw_structure"],
        "selected_protomer": {
            "source_id": _require_text(selected_protomer, "source_id"),
            "protomer_id": int(selected_protomer["protomer_id"]),
            "preprocessing_authority": _require_text(selected_protomer, "preprocessing_authority"),
            "preprocessing_backend": _require_text(selected_protomer, "preprocessing_backend"),
            "selection_method": _require_text(selected_protomer, "selection_method"),
            "rt_chain_id": _require_text(selected_protomer, "rt_chain_id"),
            "retained_context_chains": _as_string_list(selected_protomer.get("retained_context_chains")),
            "excluded_context": _as_string_list(selected_protomer.get("excluded_context")),
            "retained_context_policy": _require_text(selected_protomer, "retained_context_policy"),
        },
        "design_objectives": _require_mapping(preprocessing.get("design_objectives"), "design_objectives"),
        "chain_inventory": preprocessing["chain_inventory"],
        "acceptance_rules": preprocessing["acceptance_rules"],
        "upstream_artifact_hashes": {
            "structure_preprocessing_yaml": "sha256:" + _sha256(root / _STRUCTURE_PREPROCESSING),
            "structure_sources_yaml": "sha256:" + _sha256(root / _STRUCTURE_SOURCES),
            "residue_numbering_policy_yaml": "sha256:" + _sha256(root / _NUMBERING_POLICY),
            "backbone_bundle": "sha256:" + _sha256(backbone_bundle_path),
            "ec86kit_manifest": "sha256:" + _require_text(selected_source, "ec86kit_manifest_sha256"),
            "ec86kit_model": "sha256:" + _require_text(selected_source, "ec86kit_model_sha256"),
        },
        "numbering_policy": {
            "policy_id": numbering_policy.get("policy_id"),
            "canonical_position_basis": numbering_policy.get("canonical_position_basis"),
            "structure_position_basis": numbering_policy.get("structure_position_basis"),
            "residue_map_artifact": numbering_policy.get("residue_map_artifact"),
        },
    }
    output_path = out_root / "structure_preprocessing_manifest.yaml"
    output_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return MaterializedStructurePreprocessingArtifacts(structure_preprocessing_manifest_path=output_path)


def _validate_selected_protomer(
    *,
    selected_source: Mapping[str, Any],
    selected_protomer: Mapping[str, Any],
) -> None:
    expected_retained = _as_string_list(selected_source.get("retained_context_chains"))
    observed_retained = _as_string_list(selected_protomer.get("retained_context_chains"))
    if selected_protomer.get("source_id") != selected_source.get("source_id"):
        raise ValueError("structure preprocessing source_id must match selected structure source")
    if selected_protomer.get("rt_chain_id") != selected_source.get("rt_chain_id"):
        raise ValueError("structure preprocessing rt_chain_id must match selected structure source")
    if observed_retained != expected_retained:
        raise ValueError("structure preprocessing retained_context_chains must match selected structure source")
    if selected_protomer.get("retained_context_policy") != selected_source.get("retained_context_policy"):
        raise ValueError("structure preprocessing retained_context_policy must match selected structure source")


def _validate_chain_inventory(*, preprocessing: Mapping[str, Any], backbone_bundle: Mapping[str, Any]) -> None:
    preprocessing_rows = _chain_rows_by_id(preprocessing.get("chain_inventory"))
    backbone_rows = _chain_rows_by_id(backbone_bundle.get("chain_inventory"))
    for chain_id in ("A", "D", "E", "F"):
        preprocessing_row = preprocessing_rows.get(chain_id)
        backbone_row = backbone_rows.get(chain_id)
        if not isinstance(preprocessing_row, Mapping) or not isinstance(backbone_row, Mapping):
            raise ValueError(f"chain inventory must include selected chain {chain_id!r}")
        expected = {
            "molecule_type": backbone_row.get("molecule_type"),
            "thread_role": backbone_row.get("thread_role"),
            "retention": backbone_row.get("retention"),
        }
        observed = {
            "molecule_type": preprocessing_row.get("molecule_type"),
            "thread_role": preprocessing_row.get("thread_role"),
            "retention": preprocessing_row.get("retention"),
        }
        if observed != expected:
            raise ValueError(f"chain inventory mismatch for chain {chain_id!r}: {observed} != {expected}")


def _chain_rows_by_id(rows: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError("chain_inventory must be a list")
    return {
        str(row["chain_id"]): row
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("chain_id"), str) and row["chain_id"].strip()
    }


def _resolve_output_root(repo_root: Path, output_root: Path | None) -> Path:
    resolved = output_root or repo_root / _DEFAULT_OUTPUT_ROOT
    resolved = resolved.expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    return resolved.resolve()


def _resolve_source_ref(repo_root: Path, source_ref: str) -> Path:
    if source_ref.startswith("sibling:"):
        return (repo_root / source_ref.removeprefix("sibling:")).resolve()
    if source_ref.startswith("repo:"):
        return (repo_root / source_ref.removeprefix("repo:")).resolve()
    path = Path(source_ref).expanduser()
    return path if path.is_absolute() else (repo_root / path).resolve()


def _require_hash(path: Path, expected_sha256: str) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    observed = _sha256(path)
    expected = expected_sha256.removeprefix("sha256:")
    if observed != expected:
        raise ValueError(f"hash mismatch for {path}: {observed} != {expected}")


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


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def _find_repo_root(start: Path) -> Path:
    for parent in (start.resolve(), *start.resolve().parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("repo root with pyproject.toml not found")
