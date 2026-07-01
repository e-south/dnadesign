"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/thread_plan/pipeline.py

Materialize an explicit Eco1 RT thread-plan request from the accepted mask set.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.paths import DEFAULT_THREAD_OUTPUT_ROOT

_DOCS_ROOT = Path("docs/studies/eco1_rt_repack")
_CONTRACT_ROOT = _DOCS_ROOT / "operations/contract"
_PROFILE = _CONTRACT_ROOT / "fixtures/thread/eco1_rt_v1.profile.yaml"
_DEFAULT_OUTPUT_ROOT = DEFAULT_THREAD_OUTPUT_ROOT
_CREATED_BY = "dnadesign.studies.units.eco1_rt_repack.operations.materialization.thread_plan"
_DEFAULT_CREATED_AT = "2026-06-21T00:00:00Z"
_MASK_POLICY_ID = "eco1_rt_clade9_plurality25_direct_contact5a_v1"
_THREAD_PLAN_ARTIFACT_ID = "eco1_rt_conservative_v1.thread_plan"


@dataclass(frozen=True)
class MaterializedThreadPlanArtifacts:
    """Paths emitted by one Eco1 thread-plan materialization pass."""

    thread_plan_path: Path


def materialize_thread_plan(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    created_at: str = _DEFAULT_CREATED_AT,
    expected_mask_policy_id: str = _MASK_POLICY_ID,
    sampling_policy_overrides: Mapping[str, Any] | None = None,
    artifact_id: str = _THREAD_PLAN_ARTIFACT_ID,
) -> MaterializedThreadPlanArtifacts:
    """Materialize a planned backend request without running a backend."""

    root = (repo_root or _find_repo_root(Path.cwd())).expanduser().resolve()
    out_root = _resolve_path(root, output_root or _DEFAULT_OUTPUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    profile_path = root / _PROFILE
    profile = _load_yaml(profile_path)
    backbone_bundle_path = out_root / "backbone_bundle.yaml"
    residue_map_path = out_root / "residue_map.parquet"
    mask_set_path = out_root / "mask_set.yaml"
    for required_path in (backbone_bundle_path, residue_map_path, mask_set_path):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    mask_set = _load_yaml(mask_set_path)
    thread_plan = _build_thread_plan(
        profile=profile,
        mask_set=mask_set,
        paths={
            "profile": profile_path,
            "backbone_bundle": backbone_bundle_path,
            "residue_map": residue_map_path,
            "mask_set": mask_set_path,
        },
        created_at=created_at,
        expected_mask_policy_id=expected_mask_policy_id,
        sampling_policy_overrides=sampling_policy_overrides,
        artifact_id=artifact_id,
    )
    thread_plan_path = out_root / "thread_plan.yaml"
    thread_plan_path.write_text(yaml.safe_dump(thread_plan, sort_keys=False), encoding="utf-8")
    return MaterializedThreadPlanArtifacts(thread_plan_path=thread_plan_path)


def _build_thread_plan(
    *,
    profile: Mapping[str, Any],
    mask_set: Mapping[str, Any],
    paths: Mapping[str, Path],
    created_at: str,
    expected_mask_policy_id: str,
    sampling_policy_overrides: Mapping[str, Any] | None = None,
    artifact_id: str,
) -> dict[str, Any]:
    mask_policy_id = _require_mask_policy(mask_set, expected_mask_policy_id=expected_mask_policy_id)
    sampling_policy = dict(_require_mapping(profile.get("sampling_policy"), "sampling_policy"))
    if sampling_policy_overrides:
        sampling_policy.update(dict(sampling_policy_overrides))
    backend_kind = _selected_backend(sampling_policy)
    seed_set = _require_positive_int_list(sampling_policy.get("seed_set"), "sampling_policy.seed_set")
    temperatures = _require_positive_number_list(
        sampling_policy.get("temperatures"),
        "sampling_policy.temperatures",
    )
    batch_id = _require_text(sampling_policy, "batch_id")
    num_seq_per_target = _require_positive_int(sampling_policy.get("num_seq_per_target"), "num_seq_per_target")
    batch_size = _require_positive_int(sampling_policy.get("batch_size"), "batch_size")
    if num_seq_per_target % batch_size != 0:
        raise ValueError("sampling_policy.num_seq_per_target must be divisible by batch_size")
    fallback_policy = _require_text(sampling_policy, "backend_selection_policy")
    if fallback_policy != "explicit_no_fallback":
        raise ValueError("sampling_policy.backend_selection_policy must be explicit_no_fallback")

    rows = _require_rows(mask_set)
    fixed_positions = [int(row["canonical_position"]) for row in rows if row.get("protected") is True]
    mutable_positions = [int(row["canonical_position"]) for row in rows if row.get("non_fixed") is True]
    missing_backbone_positions = [
        int(row["canonical_position"]) for row in rows if row.get("non_fixed_missing_backbone") is True
    ]
    if not mutable_positions:
        raise ValueError("thread_plan requires at least one mapped non-fixed position")
    if set(mutable_positions) & set(missing_backbone_positions):
        raise ValueError("non_fixed_missing_backbone positions cannot be emitted as fixed-backbone mutable")

    profile_id = _require_text(profile, "profile_id")
    backend_run_id = f"{profile_id}.{backend_kind}.{mask_policy_id}.planned"
    upstream_hashes = {name: "sha256:" + _sha256(path) for name, path in paths.items()}
    source = {
        "artifact_id": _require_text(mask_set, "artifact_id"),
        "path": str(paths["mask_set"]),
        "hash": upstream_hashes["mask_set"],
        "mask_policy_id": mask_policy_id,
    }
    request_manifest = {
        "request_schema_id": "proteinmpnn.fixed_backbone_request.v1",
        "execution_status": "planned_not_run",
        "backend_kind": backend_kind,
        "backend_run_id": backend_run_id,
        "profile_id": profile_id,
        "mask_policy_id": mask_policy_id,
        "backbone_bundle_path": str(paths["backbone_bundle"]),
        "residue_map_path": str(paths["residue_map"]),
        "mask_set_path": str(paths["mask_set"]),
        "structure_chain_id": _require_nested_text(profile, ("reference", "structure_chain_id")),
        "fixed_positions": fixed_positions,
        "mutable_positions": mutable_positions,
        "excluded_positions": missing_backbone_positions,
        "excluded_position_reason": "non_fixed_missing_backbone",
        "seed_set": seed_set,
        "temperature_schedule": temperatures,
        "batch_id": batch_id,
        "num_seq_per_target": num_seq_per_target,
        "batch_size": batch_size,
        "cysteine_policy": _require_nested_text(profile, ("conservative_policy", "cysteine_policy")),
        "fallback_policy": fallback_policy,
    }
    plan_without_hash = {
        "schema_id": "thread.thread_plan",
        "schema_version": 1,
        "artifact_id": artifact_id,
        "status": "materialized",
        "created_by": _CREATED_BY,
        "created_at": created_at,
        "profile_id": profile_id,
        "mask_policy_id": mask_policy_id,
        "backend_kind": backend_kind,
        "backend_run_id": backend_run_id,
        "backend_request_manifest": request_manifest,
        "seed_set": seed_set,
        "temperature_schedule": temperatures,
        "batch_id": batch_id,
        "num_seq_per_target": num_seq_per_target,
        "batch_size": batch_size,
        "fixed_position_source": source,
        "fixed_positions": fixed_positions,
        "mutable_positions": mutable_positions,
        "excluded_non_fixed_missing_backbone_positions": missing_backbone_positions,
        "expected_sample_count": len(seed_set) * len(temperatures) * num_seq_per_target,
        "fallback_policy": fallback_policy,
        "upstream_artifact_hashes": upstream_hashes,
        "sampling_status": "pending_backend_execution",
    }
    return {"request_hash": _request_hash(plan_without_hash), **plan_without_hash}


def _selected_backend(sampling_policy: Mapping[str, Any]) -> str:
    backend = _require_text(sampling_policy, "selected_backend")
    allowed = sampling_policy.get("backends_allowed")
    if not isinstance(allowed, list) or backend not in allowed:
        raise ValueError("sampling_policy.selected_backend must be listed in backends_allowed")
    return backend


def _request_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _require_mask_policy(mask_set: Mapping[str, Any], *, expected_mask_policy_id: str) -> str:
    observed = _require_text(mask_set, "mask_policy_id")
    if observed != expected_mask_policy_id:
        raise ValueError(f"mask_set.yaml must use {expected_mask_policy_id}")
    if mask_set.get("sampling_allowed") is not True:
        raise ValueError("mask_set.yaml must allow sampling before thread_plan.yaml can be materialized")
    return observed


def _require_rows(mask_set: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = mask_set.get("residues")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("mask_set.yaml residues must be a list of mappings")
    return rows


def _require_positive_int_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise ValueError(f"{name} must contain positive integers")
        result.append(item)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate seeds")
    return result


def _require_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_positive_number_list(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result: list[float] = []
    for item in value:
        if not isinstance(item, int | float) or isinstance(item, bool) or float(item) <= 0:
            raise ValueError(f"{name} must contain positive numbers")
        result.append(float(item))
    return result


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
