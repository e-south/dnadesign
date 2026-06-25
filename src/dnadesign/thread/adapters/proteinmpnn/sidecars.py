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
