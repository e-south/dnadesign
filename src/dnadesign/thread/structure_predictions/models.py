"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/structure_predictions/models.py

Typed models for generic structure-prediction registries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StructurePredictionIssue:
    """One structure-prediction registry validation issue."""

    check_id: str
    message: str
    path: str = ""


@dataclass(frozen=True)
class StructurePredictionArtifacts:
    """Paths emitted for a structure-prediction registry."""

    registry_path: Path
