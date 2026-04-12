"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/api.py

Public latentdna execution helpers.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .src.contracts.errors import (
    AlignmentError,
    ArtifactConflictError,
    BackendUnavailableError,
    ContractViolationError,
    CoordinateSpaceError,
    MissingArtifactError,
    SourceResolutionError,
    WorkspaceValidationError,
)
from .src.services.agreement_service import compare_agreement
from .src.services.alignment_service import build_alignment
from .src.services.cluster_service import fit_cluster
from .src.services.deliverable_service import deliverable_status, list_deliverables, run_deliverable
from .src.services.distance_service import score_distance
from .src.services.enrichment_service import score_enrichment
from .src.services.export_service import export_matrix
from .src.services.inspection_service import inspect_source
from .src.services.neighbors_service import fit_neighbors
from .src.services.notebook_service import generate_notebook
from .src.services.plot_service import render_plot
from .src.services.projection_service import fit_projection
from .src.services.recipe_service import run_recipe, validate_recipe
from .src.services.sample_service import build_sample
from .src.services.scalar_service import derive_scalar
from .src.services.validation_service import validate_workspace
from .src.services.view_service import derive_view, materialize_view, reduce_view, view_stats
from .src.services.workspace_service import init_workspace, list_workspaces, show_workspace
from .src.workspaces.loader import load_workspace_config

__all__ = [
    "AlignmentError",
    "ArtifactConflictError",
    "BackendUnavailableError",
    "compare_agreement",
    "ContractViolationError",
    "CoordinateSpaceError",
    "MissingArtifactError",
    "SourceResolutionError",
    "WorkspaceValidationError",
    "build_alignment",
    "build_sample",
    "deliverable_status",
    "derive_scalar",
    "derive_view",
    "export_matrix",
    "fit_cluster",
    "score_enrichment",
    "fit_neighbors",
    "fit_projection",
    "generate_notebook",
    "init_workspace",
    "inspect_source",
    "list_deliverables",
    "list_workspaces",
    "load_workspace_config",
    "materialize_view",
    "reduce_view",
    "render_plot",
    "run_deliverable",
    "run_recipe",
    "score_distance",
    "show_workspace",
    "validate_workspace",
    "validate_recipe",
    "view_stats",
]
