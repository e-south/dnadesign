"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/manual_mask_authority/pipeline.py

Materialize Eco1 RT manual catalytic and retron-motif mask authority.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.manual_mask_authority.ontology import (
    materialize_candidate_prior_residues,
    structure_residue_ids_for_positions,
    validate_deferred_authority,
)
from dnadesign.studies.units.eco1_rt_repack.paths import DEFAULT_THREAD_OUTPUT_ROOT

_DOCS_ROOT = Path("docs/studies/eco1_rt_repack")
_CONTRACT_ROOT = _DOCS_ROOT / "operations/contract"
_PROFILE = _CONTRACT_ROOT / "fixtures/thread/eco1_rt_v1.profile.yaml"
_AUTHORITY_SOURCE = _DOCS_ROOT / "workbench/ontology/manual-mask-authority.yaml"
_DEFAULT_OUTPUT_ROOT = DEFAULT_THREAD_OUTPUT_ROOT
_CREATED_BY = "dnadesign.studies.units.eco1_rt_repack.operations.materialization.manual_mask_authority"
_DEFAULT_CREATED_AT = "2026-06-21T00:00:00Z"


@dataclass(frozen=True)
class MaterializedManualMaskAuthorityArtifacts:
    """Paths emitted by one Eco1 manual mask-authority materialization pass."""

    manual_mask_authority_path: Path


def materialize_manual_mask_authority(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    created_at: str = _DEFAULT_CREATED_AT,
) -> MaterializedManualMaskAuthorityArtifacts:
    """Materialize audited Eco1 motif anchors into a mask-authoritative artifact."""

    root = (repo_root or _find_repo_root(Path.cwd())).expanduser().resolve()
    out_root = _resolve_path(root, output_root or _DEFAULT_OUTPUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    profile_path = root / _PROFILE
    authority_source_path = root / _AUTHORITY_SOURCE
    residue_map_path = out_root / "residue_map.parquet"
    if not residue_map_path.exists():
        raise FileNotFoundError(residue_map_path)

    profile = _load_yaml(profile_path)
    authority_source = _load_yaml(authority_source_path)
    residue_rows = pq.read_table(residue_map_path).to_pylist()
    residue_by_position = {int(row["canonical_position"]): row for row in residue_rows}
    features = _materialize_features(authority_source, residue_by_position=residue_by_position)
    residue_authority_rows = _residue_authority_rows(features, residue_by_position=residue_by_position)
    candidate_prior_rows = materialize_candidate_prior_residues(
        authority_source,
        residue_by_position=residue_by_position,
    )
    deferred_authority = validate_deferred_authority(authority_source)

    manual_mask_authority = {
        "schema_id": "eco1_rt_repack.manual_mask_authority",
        "schema_version": 1,
        "artifact_id": "eco1_rt_conservative_v1.manual_mask_authority",
        "status": "materialized",
        "created_by": _CREATED_BY,
        "created_at": created_at,
        "profile_id": _require_text(profile, "profile_id"),
        "coordinate_space": _require_text(authority_source, "coordinate_space"),
        "target_row_id": _require_text(authority_source, "target_row_id"),
        "target_sequence_hash": _require_text(authority_source, "target_sequence_hash"),
        "mask_policy_id": _require_text(authority_source, "mask_policy_id"),
        "authority_source_ref": f"repo:{_AUTHORITY_SOURCE}",
        "upstream_artifact_hashes": {
            "profile": "sha256:" + _sha256(profile_path),
            "manual_mask_authority_source": "sha256:" + _sha256(authority_source_path),
            "residue_map": "sha256:" + _sha256(residue_map_path),
        },
        "summary": {
            "protected_feature_count": _protected_feature_count(features),
            "rt_interval_feature_count": _rt_interval_feature_count(features),
            "manual_mask_position_count": len(residue_authority_rows),
            "candidate_prior_position_count": len(candidate_prior_rows),
            "context_only_span_count": len(_as_list(authority_source.get("context_only_spans"), "context_only_spans")),
            "deferred_authority_count": len(deferred_authority),
        },
        "source_basis": _as_list(authority_source.get("source_basis"), "source_basis"),
        "features": features,
        "residues": residue_authority_rows,
        "candidate_prior_residues": candidate_prior_rows,
        "context_only_spans": _as_list(authority_source.get("context_only_spans"), "context_only_spans"),
        "deferred_authority": deferred_authority,
    }
    output_path = out_root / "manual_mask_authority.yaml"
    output_path.write_text(yaml.safe_dump(manual_mask_authority, sort_keys=False), encoding="utf-8")
    return MaterializedManualMaskAuthorityArtifacts(manual_mask_authority_path=output_path)


def _materialize_features(
    authority_source: Mapping[str, Any],
    *,
    residue_by_position: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for authority_set in _as_list(authority_source.get("authority_sets"), "authority_sets"):
        set_id = _require_text(authority_set, "id")
        for feature in _as_list(authority_set.get("features"), f"authority_sets[{set_id}].features"):
            start = _require_int(feature, "start")
            end = _require_int(feature, "end")
            if start > end:
                raise ValueError(f"manual mask feature {feature.get('id')!r} has start after end")
            canonical_positions = list(range(start, end + 1))
            for position in canonical_positions:
                if position not in residue_by_position:
                    raise ValueError(
                        f"manual mask feature {feature.get('id')!r} references unknown position {position}"
                    )
            _validate_feature_sequence(
                feature_id=_require_text(feature, "id"),
                authority_type=_require_text(authority_set, "authority_type"),
                canonical_positions=canonical_positions,
                residue_by_position=residue_by_position,
            )
            source_locator = _require_text(feature, "source_locator")
            features.append(
                {
                    "feature_id": _require_text(feature, "id"),
                    "authority_set_id": set_id,
                    "authority_type": _require_text(authority_set, "authority_type"),
                    "label": _require_text(feature, "label"),
                    "policy": _require_text(authority_set, "policy"),
                    "reason": _require_text(feature, "reason"),
                    "canonical_start": start,
                    "canonical_end": end,
                    "canonical_positions": canonical_positions,
                    "structure_residue_ids": structure_residue_ids_for_positions(
                        canonical_positions,
                        residue_by_position=residue_by_position,
                    ),
                    "source_locator": source_locator,
                    "evidence_basis": _as_list(feature.get("evidence_basis"), "evidence_basis"),
                }
            )
    if not features:
        raise ValueError("manual mask authority source must declare at least one feature")
    return features


def _validate_feature_sequence(
    *,
    feature_id: str,
    authority_type: str,
    canonical_positions: Sequence[int],
    residue_by_position: Mapping[int, Mapping[str, Any]],
) -> None:
    sequence = "".join(str(residue_by_position[position]["wt_aa"]) for position in canonical_positions)
    if authority_type == "retron_x_motif_anchor" and not (
        len(sequence) == 5 and sequence.startswith("NA") and sequence.endswith("H")
    ):
        raise ValueError(
            f"manual mask feature {feature_id!r} must resolve to an EC86 NAxxH motif, observed {sequence!r}"
        )
    if authority_type == "catalytic_core_motif_anchor" and sequence != "YADD":
        raise ValueError(f"manual mask feature {feature_id!r} must resolve to EC86 YADD, observed {sequence!r}")
    if authority_type == "retron_y_motif_anchor" and sequence != "VTG":
        raise ValueError(f"manual mask feature {feature_id!r} must resolve to EC86 VTG, observed {sequence!r}")


def _residue_authority_rows(
    features: Sequence[Mapping[str, Any]],
    *,
    residue_by_position: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    features_by_position: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for feature in features:
        for position in feature["canonical_positions"]:
            features_by_position[int(position)].append(feature)

    rows: list[dict[str, Any]] = []
    for position in sorted(features_by_position):
        residue = residue_by_position[position]
        feature_rows = [feature for feature in features_by_position[position] if feature.get("policy") == "fixed"]
        if not feature_rows:
            continue
        if residue.get("mapping_status") != "mapped":
            raise ValueError(f"manual mask position {position} is not mapped in residue_map.parquet")
        reasons = sorted({str(feature["reason"]) for feature in feature_rows})
        rows.append(
            {
                "canonical_position": position,
                "wt_aa": residue["wt_aa"],
                "structure_chain_id": residue["structure_chain_id"],
                "structure_residue_id": residue["structure_residue_id"],
                "design_position": residue["design_position"],
                "mapping_status": residue["mapping_status"],
                "manual_mask": True,
                "manual_mask_reason": ";".join(reasons),
                "authority_feature_ids": sorted(str(feature["feature_id"]) for feature in feature_rows),
                "authority_set_ids": sorted(str(feature["authority_set_id"]) for feature in feature_rows),
                "policy": "fixed",
            }
        )
    return rows


def _protected_feature_count(features: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for feature in features if feature.get("policy") == "fixed")


def _rt_interval_feature_count(features: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for feature in features if feature.get("authority_type") == "rt_core_interval")


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


def _as_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _require_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _require_int(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _find_repo_root(start: Path) -> Path:
    for parent in (start.resolve(), *start.resolve().parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("repo root with pyproject.toml not found")
