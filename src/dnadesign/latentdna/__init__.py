"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/__init__.py

Public latentdna package exports.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    "AlignmentError": (".api", "AlignmentError"),
    "ArtifactConflictError": (".api", "ArtifactConflictError"),
    "ArtifactManifest": (".src.contracts.manifest", "ArtifactManifest"),
    "BackendUnavailableError": (".api", "BackendUnavailableError"),
    "CommandResult": (".src.contracts.result", "CommandResult"),
    "DeliverableStatusResult": (".src.contracts.deliverable", "DeliverableStatusResult"),
    "ContractViolationError": (".contracts", "ContractViolationError"),
    "CoordinateSpaceError": (".api", "CoordinateSpaceError"),
    "MissingArtifactError": (".api", "MissingArtifactError"),
    "SourceResolutionError": (".api", "SourceResolutionError"),
    "WorkspaceConfig": (".src.contracts.workspace", "WorkspaceConfig"),
    "WorkspaceContext": (".src.workspaces.loader", "WorkspaceContext"),
    "WorkspaceValidationError": (".api", "WorkspaceValidationError"),
    "build_alignment": (".api", "build_alignment"),
    "build_sample": (".api", "build_sample"),
    "deliverable_status": (".api", "deliverable_status"),
    "derive_scalar": (".api", "derive_scalar"),
    "derive_view": (".api", "derive_view"),
    "export_matrix": (".api", "export_matrix"),
    "fit_cluster": (".api", "fit_cluster"),
    "fit_projection": (".api", "fit_projection"),
    "generate_notebook": (".api", "generate_notebook"),
    "init_workspace": (".api", "init_workspace"),
    "inspect_source": (".api", "inspect_source"),
    "list_deliverables": (".api", "list_deliverables"),
    "list_workspaces": (".api", "list_workspaces"),
    "load_workspace_config": (".api", "load_workspace_config"),
    "materialize_view": (".api", "materialize_view"),
    "reduce_view": (".api", "reduce_view"),
    "render_plot": (".api", "render_plot"),
    "run_deliverable": (".api", "run_deliverable"),
    "run_recipe": (".api", "run_recipe"),
    "score_distance": (".api", "score_distance"),
    "score_enrichment": (".api", "score_enrichment"),
    "show_workspace": (".api", "show_workspace"),
    "validate_recipe": (".api", "validate_recipe"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
