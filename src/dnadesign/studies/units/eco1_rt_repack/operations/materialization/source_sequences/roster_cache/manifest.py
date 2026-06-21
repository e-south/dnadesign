"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/roster_cache/manifest.py

Manifest writer for Eco1 conservation roster caches.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

_SCHEMA_ID = "eco1_rt_repack.conservation_source_sequence_cache.index"


def write_roster_cache_manifest(
    path: Path,
    *,
    roster_table: Path,
    roster_table_sha256: str,
    conservation_sources_sha256: str,
    source_records_path: Path,
    provider_cache_hashes: Mapping[str, str],
    profile_counts: Mapping[str, Mapping[str, int]],
    roster_hash_policy: str,
    created_at: str,
) -> None:
    """Write the roster-cache index manifest."""

    payload: dict[str, Any] = {
        "schema_id": _SCHEMA_ID,
        "schema_version": 1,
        "status": "materialized",
        "roster_table": str(roster_table),
        "roster_table_sha256": roster_table_sha256,
        "roster_hash_policy": roster_hash_policy,
        "source_records_path": str(source_records_path),
        "provider_cache_hashes": dict(provider_cache_hashes),
        "upstream_hashes": {
            "conservation_sources_yaml": conservation_sources_sha256,
            "roster_table": roster_table_sha256,
        },
        "profile_counts": {profile_id: dict(counts) for profile_id, counts in profile_counts.items()},
        "created_at": created_at,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
