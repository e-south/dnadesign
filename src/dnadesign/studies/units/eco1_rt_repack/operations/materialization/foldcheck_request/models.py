"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_request/models.py

Typed models for Eco1 fold-check request materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MaterializedFoldCheckRequestArtifacts:
    """Paths emitted by one Eco1 fold-check request materialization pass."""

    input_fasta_path: Path
    request_manifest_path: Path
