"""Manifest helpers for Eco1 conservation alignment bundles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

_INDEX_SCHEMA_ID = "eco1_rt_repack.conservation_alignment_bundle.index"


def write_alignment_index_manifest(
    path: Path,
    *,
    profile_ids: Sequence[str],
    alignment_manifests: Mapping[str, Path],
    aligned_fasta_paths: Mapping[str, Path],
    source_fasta_paths: Mapping[str, Path],
    target_row_id: str,
    target_sequence_hash: str,
    command_args: Sequence[str],
    profile_runs: Sequence[Mapping[str, Any]],
    upstream_hashes: Mapping[str, str],
    total_elapsed_seconds: float,
    created_at: str,
) -> None:
    """Write the bundle-level aligned FASTA index manifest."""

    payload = {
        "schema_id": _INDEX_SCHEMA_ID,
        "schema_version": 1,
        "status": "materialized",
        "profile_ids": list(profile_ids),
        "alignment_manifests": {profile_id: str(path) for profile_id, path in alignment_manifests.items()},
        "aligned_fasta_paths": {profile_id: str(path) for profile_id, path in aligned_fasta_paths.items()},
        "source_fasta_paths": {profile_id: str(path) for profile_id, path in source_fasta_paths.items()},
        "target_row_id": target_row_id,
        "target_sequence_hash": target_sequence_hash,
        "command_args": list(command_args),
        "profile_runs": list(profile_runs),
        "upstream_hashes": dict(upstream_hashes),
        "total_elapsed_seconds": total_elapsed_seconds,
        "created_at": created_at,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
