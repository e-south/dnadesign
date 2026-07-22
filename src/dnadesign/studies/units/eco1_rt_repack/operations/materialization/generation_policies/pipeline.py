"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/generation_policies/pipeline.py

Materialize Eco1 RT generation-policy manifests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.paths import (
    find_repo_root,
    write_yaml,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.alphabet_policy import (
    build_alphabet_rows,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.config import (
    build_default_generation_policy_config,
    validate_generation_policy_config,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    CREATED_BY,
    DEFAULT_CREATED_AT,
    DEFAULT_GENERATION_POLICIES_ROOT,
    DEFAULT_SOURCE_OUTPUT_ROOT,
    GENERATION_POLICY_VERSION,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.models import (
    GenerationPolicyConfig,
    MaterializedGenerationPolicies,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.position_policy import (
    build_position_rows,
)
from dnadesign.thread.adapters.proteinmpnn.hashing import sha256_uri


def materialize_generation_policies(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    source_output_root: Path | None = None,
    config: Mapping[str, Any] | None = None,
    created_at: str = DEFAULT_CREATED_AT,
) -> MaterializedGenerationPolicies:
    """Materialize active generation-policy manifests without running ProteinMPNN."""

    root = (repo_root or find_repo_root(Path.cwd())).expanduser().resolve()
    out_root = _resolve_path(root, output_root or DEFAULT_GENERATION_POLICIES_ROOT)
    source_root = _resolve_path(root, source_output_root or DEFAULT_SOURCE_OUTPUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    validated_config = validate_generation_policy_config(config or build_default_generation_policy_config())
    inputs = _load_inputs(source_root)
    position_rows = build_position_rows(config=validated_config, inputs=inputs)
    alphabet_rows = build_alphabet_rows(
        config=validated_config,
        position_rows=position_rows,
        conservation_rows=inputs["conservation_rows"],
        source_root=source_root,
    )

    positions_path = out_root / "generation_policy_positions.parquet"
    alphabets_path = out_root / "generation_policy_alphabets.parquet"
    _write_parquet(positions_path, position_rows)
    _write_parquet(alphabets_path, alphabet_rows)

    manifest_without_hash = _build_manifest(
        config=validated_config,
        created_at=created_at,
        source_root=source_root,
        output_root=out_root,
        input_hashes={
            "residue_map": sha256_uri(source_root / "residue_map.parquet"),
            "contact_geometry_profile": sha256_uri(source_root / "contact_geometry_profile.parquet"),
            "conservation_profile": sha256_uri(source_root / "conservation_profile.parquet"),
        },
        position_rows=position_rows,
        alphabet_rows=alphabet_rows,
        positions_path=positions_path,
        alphabets_path=alphabets_path,
    )
    manifest = {"policy_manifest_hash": _payload_hash(manifest_without_hash), **manifest_without_hash}
    manifest_path = out_root / "generation_policy_manifest.yaml"
    write_yaml(manifest_path, manifest)
    return MaterializedGenerationPolicies(
        manifest_path=manifest_path,
        positions_path=positions_path,
        alphabets_path=alphabets_path,
    )


def _load_inputs(source_root: Path) -> dict[str, list[dict[str, Any]]]:
    paths = {
        "residue_rows": source_root / "residue_map.parquet",
        "contact_rows": source_root / "contact_geometry_profile.parquet",
        "conservation_rows": source_root / "conservation_profile.parquet",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    return {key: pq.read_table(path).to_pylist() for key, path in paths.items()}


def _build_manifest(
    *,
    config: GenerationPolicyConfig,
    created_at: str,
    source_root: Path,
    output_root: Path,
    input_hashes: Mapping[str, str],
    position_rows: list[dict[str, Any]],
    alphabet_rows: list[dict[str, Any]],
    positions_path: Path,
    alphabets_path: Path,
) -> dict[str, Any]:
    position_rows_by_policy = _group_rows_by_policy(position_rows)
    alphabet_rows_by_policy = _group_rows_by_policy(alphabet_rows)
    return {
        "schema_id": "eco1_rt.generation_policy_manifest",
        "schema_version": 1,
        "status": "materialized",
        "created_by": CREATED_BY,
        "created_at": created_at,
        "generation_policy_version": GENERATION_POLICY_VERSION,
        "source_output_root": str(source_root),
        "generation_policies_root": str(output_root),
        "position_manifest_path": str(positions_path),
        "alphabet_manifest_path": str(alphabets_path),
        "upstream_artifact_hashes": dict(input_hashes),
        "generation_total_target_raw": config.generation_total_target_raw,
        "generation_policies": [
            {
                "policy_id": policy.policy_id,
                "policy_version": GENERATION_POLICY_VERSION,
                "open_set_id": policy.open_set_id,
                "alphabet_rule_id": policy.alphabet_rule_id,
                "requested_variants": policy.requested_variants,
                "purpose": policy.purpose,
                "open_position_count": sum(
                    1 for row in position_rows_by_policy[policy.policy_id] if row["is_open_position"]
                ),
                "protected_position_count": sum(
                    1 for row in position_rows_by_policy[policy.policy_id] if row["protected_reason_codes"]
                ),
                "alphabet_enforcement_modes": sorted(
                    {row["alphabet_enforcement_mode"] for row in alphabet_rows_by_policy[policy.policy_id]}
                ),
            }
            for policy in config.enabled_policies
        ],
        "conceptual_boundary": (
            "ProteinMPNN outputs are complete sequences sampled under one policy; do not combine mutations "
            "from separate policy outputs after generation."
        ),
    }


def _group_rows_by_policy(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_policy.setdefault(str(row["policy_id"]), []).append(row)
    return by_policy


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _resolve_path(repo_root: Path, path: Path) -> Path:
    resolved = path.expanduser()
    return resolved if resolved.is_absolute() else (repo_root / resolved).resolve()


def _payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def generation_policy_payload_hash(payload: Mapping[str, Any]) -> str:
    """Return the stable hash used by generation-policy manifests."""

    return _payload_hash(payload)
