"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/surface_accessibility/models.py

Typed fragments for Eco1 RT surface-accessibility materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MaterializedSurfaceAccessibilityArtifacts:
    """Paths emitted by the surface-accessibility materializer."""

    surface_accessibility_profile_path: Path
