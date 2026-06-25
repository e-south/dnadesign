"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/proteinmpnn/sidecars.py

ProteinMPNN helper JSONL sidecar construction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def assigned_chains_payload(*, target_name: str, chain_id: str) -> dict[str, list[list[str]]]:
    """Build the official helper-compatible fixed-chain assignment payload."""

    return {target_name: [[chain_id], []]}


def fixed_positions_payload(
    *, target_name: str, chain_id: str, fixed_positions: list[int]
) -> dict[str, dict[str, list[int]]]:
    """Build the official helper-compatible fixed-position payload."""

    return {target_name: {chain_id: fixed_positions}}


def write_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a single-record JSONL helper sidecar."""

    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def resolve_manifest_sidecar_path(manifest_path: Path, value: Any) -> Path:
    """Resolve a sidecar path recorded before a request manifest was copied."""

    recorded = Path(str(value))
    colocated = manifest_path.parent / recorded.name
    if recorded.name and colocated.exists():
        return colocated
    if recorded.is_absolute():
        return recorded
    return manifest_path.parent / recorded


def resolve_manifest_sidecar_paths(manifest_path: Path, sidecar_paths: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve every sidecar declared by a copied ProteinMPNN request manifest."""

    return {str(name): resolve_manifest_sidecar_path(manifest_path, value) for name, value in sidecar_paths.items()}
