"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/proteinmpnn/positions.py

Position-basis conversion for ProteinMPNN request adapters.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def mapped_chain_rows(
    residue_map_path: Path,
    *,
    chain_id: str,
    expected_mapped_count: int | None = None,
) -> list[dict[str, Any]]:
    """Return canonical rows with fixed-backbone coordinates for one chain."""

    rows = pq.read_table(residue_map_path).to_pylist()
    mapped = [
        row for row in rows if row.get("mapping_status") == "mapped" and row.get("structure_chain_id") == chain_id
    ]
    mapped.sort(key=lambda row: int(row["canonical_position"]))
    if expected_mapped_count is not None and len(mapped) != expected_mapped_count:
        raise ValueError(f"ProteinMPNN request requires {expected_mapped_count} mapped chain-{chain_id} residues")
    return mapped


def to_proteinmpnn_positions(value: Any, mapping: Mapping[int, int], name: str) -> list[int]:
    """Convert canonical positions to ProteinMPNN chain-local 1-indexed positions."""

    canonical_positions = require_int_list(value, name)
    converted: list[int] = []
    for position in canonical_positions:
        if position not in mapping:
            raise ValueError(f"{name} contains canonical position {position} without fixed-backbone coordinates")
        converted.append(mapping[position])
    return sorted(converted)


def require_int_list(value: Any, name: str) -> list[int]:
    """Require a duplicate-free list of integer positions."""

    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    result: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"{name} must contain integers")
        result.append(item)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def require_missing_backbone_excluded(excluded_positions: list[int], mapping: Mapping[int, int]) -> None:
    """Fail if a missing-backbone exclusion unexpectedly has fixed-backbone coordinates."""

    leaked = sorted(set(excluded_positions) & set(mapping))
    if leaked:
        raise ValueError(f"excluded missing-backbone positions unexpectedly mapped: {leaked}")
