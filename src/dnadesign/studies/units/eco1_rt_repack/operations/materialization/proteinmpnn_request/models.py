"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/proteinmpnn_request/models.py

Typed result models for Eco1 ProteinMPNN request materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MaterializedProteinMpnnRequestArtifacts:
    """Paths emitted by one ProteinMPNN request materialization pass."""

    chain_a_backbone_pdb_path: Path
    parsed_pdbs_path: Path
    assigned_chains_path: Path
    fixed_positions_path: Path
    request_manifest_path: Path
