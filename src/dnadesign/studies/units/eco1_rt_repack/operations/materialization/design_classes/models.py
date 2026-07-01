"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/design_classes/models.py

Typed value objects for Eco1 RT design-class expansion materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DesignClassSpec:
    """One named mask/sampling class for fixed-backbone Eco1 RT repacking."""

    design_class_id: str
    path_id: str
    role: str
    conservation_profile_id: str
    conservation_threshold: float
    contact_threshold_angstrom: float
    batch_id: str
    premise: str
    rationale: str


@dataclass(frozen=True)
class DesignClassArtifact:
    """Paths emitted for one generated design class."""

    design_class_id: str
    class_root: Path
    mask_set_path: Path
    thread_plan_path: Path
    request_manifest_path: Path


@dataclass(frozen=True)
class MaterializedDesignClassRequests:
    """Paths emitted by design-class request materialization."""

    manifest_path: Path
    class_artifacts: tuple[DesignClassArtifact, ...]


@dataclass(frozen=True)
class MaterializedDesignClassCandidatePool:
    """Paths emitted by design-class candidate-pool materialization."""

    candidate_pool_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class MaterializedDesignClassFoldCheckRequest:
    """Paths emitted by expanded fold-check request materialization."""

    input_fasta_path: Path
    request_manifest_path: Path
