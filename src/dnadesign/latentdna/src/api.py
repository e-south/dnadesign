"""
Public latentdna execution helpers.
"""

from __future__ import annotations

from .contracts.errors import (
    AlignmentError,
    ArtifactConflictError,
    BackendUnavailableError,
    ContractViolationError,
    MemoryPreflightError,
    MissingArtifactError,
    SourceResolutionError,
)
from .services.agreement_service import compare_agreement
from .services.alignment_service import build_alignment
from .services.catalog_service import explain_deliverable, explain_export, explain_plot, explain_workspace
from .services.cluster_service import fit_cluster
from .services.deliverable_service import deliverable_status, list_deliverables, run_deliverable
from .services.distance_service import score_distance
from .services.enrichment_service import score_enrichment
from .services.export_service import export_matrix
from .services.inspection_service import inspect_source
from .services.neighbors_service import fit_neighbors
from .services.notebook_service import generate_notebook
from .services.plot_service import render_plot
from .services.projection_service import fit_projection
from .services.recipe_service import run_recipe, validate_recipe
from .services.sample_service import build_sample
from .services.scalar_service import derive_scalar
from .services.validation_service import validate_workspace
from .services.view_service import derive_view, materialize_view, reduce_view, view_stats
from .services.workspace_service import init_workspace, list_workspaces, show_workspace

__all__ = [
    "AlignmentError",
    "ArtifactConflictError",
    "BackendUnavailableError",
    "compare_agreement",
    "ContractViolationError",
    "MemoryPreflightError",
    "MissingArtifactError",
    "SourceResolutionError",
    "build_alignment",
    "build_sample",
    "deliverable_status",
    "derive_scalar",
    "derive_view",
    "explain_deliverable",
    "explain_export",
    "explain_plot",
    "explain_workspace",
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
