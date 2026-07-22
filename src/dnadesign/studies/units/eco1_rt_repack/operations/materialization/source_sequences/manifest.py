"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/manifest.py

Manifest helpers for Eco1 conservation source-sequence bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

_PROFILE_SCHEMA_ID = "eco1_rt_repack.conservation_source_sequence_bundle.profile"
_INDEX_SCHEMA_ID = "eco1_rt_repack.conservation_source_sequence_bundle.index"


def write_profile_manifest(
    path: Path,
    *,
    profile_id: str,
    fasta_path: Path,
    fasta_sha256: str,
    target_row_id: str,
    target_sequence_hash: str,
    included_records: Sequence[Mapping[str, Any]],
    excluded_records: Sequence[Mapping[str, Any]],
    upstream_hashes: Mapping[str, str],
    created_at: str,
) -> None:
    """Write one profile-level source FASTA manifest."""

    payload = {
        "schema_id": _PROFILE_SCHEMA_ID,
        "schema_version": 1,
        "status": "materialized",
        "profile_id": profile_id,
        "fasta_path": str(fasta_path),
        "fasta_sha256": fasta_sha256,
        "target_row_id": target_row_id,
        "target_sequence_hash": target_sequence_hash,
        "included_record_count": len(included_records),
        "excluded_record_count": len(excluded_records),
        "included_records": list(included_records),
        "excluded_records": list(excluded_records),
        "upstream_hashes": dict(upstream_hashes),
        "created_at": created_at,
    }
    _write_yaml(path, payload)


def write_index_manifest(
    path: Path,
    *,
    profile_ids: Sequence[str],
    profile_manifests: Mapping[str, Path],
    target_row_id: str,
    target_sequence_hash: str,
    upstream_hashes: Mapping[str, str],
    created_at: str,
) -> None:
    """Write the bundle-level source FASTA index manifest."""

    payload = {
        "schema_id": _INDEX_SCHEMA_ID,
        "schema_version": 1,
        "status": "materialized",
        "profile_ids": list(profile_ids),
        "profile_manifests": {profile_id: str(path) for profile_id, path in profile_manifests.items()},
        "target_row_id": target_row_id,
        "target_sequence_hash": target_sequence_hash,
        "upstream_hashes": dict(upstream_hashes),
        "created_at": created_at,
    }
    _write_yaml(path, payload)


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")
