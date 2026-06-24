"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/mask_set/pipeline.py

Materialize Eco1 RT residue mask sets from contact and conservation evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.masking import (
    compose_mask_rows,
    summarize_mask_rows,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.manual_mask_authority import (
    materialize_manual_mask_authority,
)

_DOCS_ROOT = Path("docs/studies/eco1_rt_repack")
_CONTRACT_ROOT = _DOCS_ROOT / "operations/contract"
_PROFILE = _CONTRACT_ROOT / "fixtures/thread/eco1_rt_v1.profile.yaml"
_CONSERVATION_SOURCES = _DOCS_ROOT / "workbench/provenance/conservation-sources.yaml"
_DEFAULT_OUTPUT_ROOT = Path("outputs/thread/eco1_rt_conservative_v1")
_CREATED_BY = "dnadesign.studies.units.eco1_rt_repack.operations.materialization.mask_set"
_DEFAULT_CREATED_AT = "2026-06-21T00:00:00Z"


@dataclass(frozen=True)
class MaterializedMaskSetArtifacts:
    """Paths emitted by one Eco1 mask-set materialization pass."""

    manual_mask_authority_path: Path
    mask_set_path: Path


def materialize_mask_set(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    created_at: str = _DEFAULT_CREATED_AT,
) -> MaterializedMaskSetArtifacts:
    """Materialize the conservative Eco1 mask set from existing evidence profiles."""

    root = (repo_root or _find_repo_root(Path.cwd())).expanduser().resolve()
    out_root = _resolve_path(root, output_root or _DEFAULT_OUTPUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    profile = _load_yaml(root / _PROFILE)
    conservation_sources = _load_yaml(root / _CONSERVATION_SOURCES)
    residue_map_path = out_root / "residue_map.parquet"
    contact_geometry_profile_path = out_root / "contact_geometry_profile.parquet"
    conservation_profile_path = out_root / "conservation_profile.parquet"
    for required_path in (residue_map_path, contact_geometry_profile_path, conservation_profile_path):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    residue_rows = pq.read_table(residue_map_path).to_pylist()
    contact_geometry_rows = pq.read_table(contact_geometry_profile_path).to_pylist()
    conservation_rows = pq.read_table(conservation_profile_path).to_pylist()
    manual_authority_result = materialize_manual_mask_authority(
        repo_root=root,
        output_root=out_root,
        created_at=created_at,
    )
    manual_authority = _load_yaml(manual_authority_result.manual_mask_authority_path)
    selected_rows = compose_mask_rows(
        residue_rows=residue_rows,
        contact_geometry_rows=contact_geometry_rows,
        conservation_rows=conservation_rows,
        manual_authority=manual_authority,
    )

    mask_set_path = out_root / "mask_set.yaml"
    mask_set = _build_mask_set(
        rows=selected_rows,
        profile=profile,
        conservation_sources=conservation_sources,
        upstream_hashes={
            "profile": "sha256:" + _sha256(root / _PROFILE),
            "conservation_sources": "sha256:" + _sha256(root / _CONSERVATION_SOURCES),
            "residue_map": "sha256:" + _sha256(residue_map_path),
            "contact_geometry_profile": "sha256:" + _sha256(contact_geometry_profile_path),
            "conservation_profile": "sha256:" + _sha256(conservation_profile_path),
            "manual_mask_authority": "sha256:" + _sha256(manual_authority_result.manual_mask_authority_path),
        },
        created_at=created_at,
    )
    mask_set_path.write_text(yaml.safe_dump(mask_set, sort_keys=False), encoding="utf-8")
    return MaterializedMaskSetArtifacts(
        manual_mask_authority_path=manual_authority_result.manual_mask_authority_path,
        mask_set_path=mask_set_path,
    )


def _build_mask_set(
    *,
    rows: list[dict[str, Any]],
    profile: Mapping[str, Any],
    conservation_sources: Mapping[str, Any],
    upstream_hashes: Mapping[str, str],
    created_at: str,
) -> dict[str, Any]:
    non_fixed_mapped_count = sum(1 for row in rows if row["non_fixed"])
    return {
        "schema_id": "thread.mask_set",
        "schema_version": 1,
        "artifact_id": "eco1_rt_conservative_v1.mask_set",
        "status": "materialized",
        "created_by": _CREATED_BY,
        "created_at": created_at,
        "profile_id": _require_text(profile, "profile_id"),
        "mask_policy_id": "eco1_rt_clade9_plurality25_direct_contact5a_v1",
        "sampling_status": (
            "blocked_no_non_fixed_mapped_positions" if non_fixed_mapped_count == 0 else "pending_sampling_plan"
        ),
        "sampling_allowed": non_fixed_mapped_count > 0,
        "manual_mask_authority_status": "materialized_eco1_rt_manual_motif_wang_direct_contact_v1",
        "cysteine_policy": "no_new_cysteine_candidate_ingest",
        "source_method_id": _require_nested_text(conservation_sources, ("source_method", "method_id")),
        "upstream_artifact_hashes": dict(upstream_hashes),
        "summary": summarize_mask_rows(rows),
        "residues": rows,
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


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_nested_text(payload: Mapping[str, Any], fields: Sequence[str]) -> str:
    current: Any = payload
    for field in fields:
        current = _require_mapping(current, ".".join(fields)).get(field)
    if not isinstance(current, str) or not current.strip():
        raise ValueError(f"{'.'.join(fields)} must be a non-empty string")
    return current.strip()


def _require_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _find_repo_root(start: Path) -> Path:
    for parent in (start.resolve(), *start.resolve().parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("repo root with pyproject.toml not found")
