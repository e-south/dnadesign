"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_review/models.py

Typed models for Eco1 fold-check review materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MaterializedFoldCheckReviewArtifacts:
    """Paths emitted by one Eco1 fold-check review materialization pass."""

    ranking_path: Path
    structure_panel_path: Path
    full_structure_set_path: Path
    atlas_subset_manifest_path: Path
    chimerax_script_path: Path
    full_chimerax_script_path: Path
    visual_manifest_path: Path
    notebook_path: Path
    selected_structure_count: int
    full_structure_count: int
    plot_count: int


@dataclass(frozen=True)
class PanelEntry:
    """One structure selected for review."""

    candidate_id: str
    selection_stratum: str
    source_model_artifact_path: str
    local_model_artifact_path: str
    copy_status: str
    source_model_artifact_hash: str
    display_label: str = ""
    sequence_identity_percent: float | None = None
    proteinmpnn_rank: int | None = None
    wt_runtime_ca_rmsd: float | None = None
