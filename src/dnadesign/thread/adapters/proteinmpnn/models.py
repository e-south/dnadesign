"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/proteinmpnn/models.py

Typed models for generic ProteinMPNN request adaptation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProteinMpnnBackboneExport:
    """ProteinMPNN backbone export and parsed JSON payload."""

    parsed_payload: dict[str, Any]
    canonical_to_proteinmpnn_position: dict[int, int]


@dataclass(frozen=True)
class ProteinMpnnRequestIssue:
    """Generic ProteinMPNN request validation issue."""

    check_id: str
    message: str
    path: str


@dataclass(frozen=True)
class ProteinMpnnRequestArtifacts:
    """Paths emitted by one ProteinMPNN request adaptation pass."""

    backbone_pdb_path: Path
    parsed_pdbs_path: Path
    assigned_chains_path: Path
    fixed_positions_path: Path
    request_manifest_path: Path


@dataclass(frozen=True)
class ProteinMpnnRunArtifacts:
    """Paths emitted by one ProteinMPNN backend execution pass."""

    backend_run_manifest_path: Path
    sample_table_path: Path
