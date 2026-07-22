"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/twist_handoff/models.py

Result models for the Eco1 RT Twist handoff.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MaterializedTwistHandoff:
    """Paths emitted by one validated Twist handoff materialization."""

    manifest_path: Path
    twist_csv_path: Path
    fasta_path: Path
    genbank_paths: tuple[Path, ...]
