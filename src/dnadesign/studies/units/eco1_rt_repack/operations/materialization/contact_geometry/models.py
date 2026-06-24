"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact_geometry/models.py

Data contracts for Eco1 RT contact-geometry materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MaterializedContactGeometryArtifacts:
    """Paths emitted by one Eco1 contact-geometry materialization pass."""

    contact_geometry_profile_path: Path


@dataclass(frozen=True)
class AtomSite:
    """Minimal atom metadata needed for retained-context geometry."""

    coord: np.ndarray
    chain_id: str
    molecule_type: str
    residue_id: int
    residue_name: str
    atom_name: str


@dataclass(frozen=True)
class NearestAtomResult:
    """Nearest retained-context atom result for one residue atom class."""

    distance: float | None
    atom: AtomSite | None
