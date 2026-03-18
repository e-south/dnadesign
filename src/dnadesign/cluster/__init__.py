"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/__init__.py

Public cluster package exports.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .api import (
        ClusterApiError,  # noqa: F401
        ClusterExecutionResult,  # noqa: F401
        list_runs,  # noqa: F401
        list_workspace_runs,  # noqa: F401
        run_analyze,  # noqa: F401
        run_analyze_workspace,  # noqa: F401
        run_fit,  # noqa: F401
        run_fit_workspace,  # noqa: F401
        run_sweep,  # noqa: F401
        run_sweep_workspace,  # noqa: F401
        run_umap,  # noqa: F401
        run_umap_workspace,  # noqa: F401
    )
    from .src.analysis.contracts import AnalysisRequest  # noqa: F401
    from .src.methods import (
        ClusteringMethod,  # noqa: F401
        MethodRegistry,  # noqa: F401
        default_method_registry,  # noqa: F401
        register_method,  # noqa: F401
        registered_methods,  # noqa: F401
        supported_method_ids,  # noqa: F401
    )
    from .src.runs.contracts import AnalysisRun, ClusterRun, EmbeddingRun, RunCounts  # noqa: F401
    from .src.runtime_contracts import FeatureSpec, FitRequest, InputSource, MethodConfig  # noqa: F401
    from .src.workspaces import (  # noqa: F401
        WorkspaceConfig,
        builtin_workspaces_dir,
        init_workspace,
        load_workspace_config,
    )

_EXPORTS: dict[str, tuple[str, str]] = {
    "AnalysisRequest": (".src.analysis.contracts", "AnalysisRequest"),
    "AnalysisRun": (".src.runs.contracts", "AnalysisRun"),
    "ClusterApiError": (".api", "ClusterApiError"),
    "ClusterExecutionResult": (".api", "ClusterExecutionResult"),
    "ClusterRun": (".src.runs.contracts", "ClusterRun"),
    "ClusteringMethod": (".src.methods", "ClusteringMethod"),
    "MethodRegistry": (".src.methods", "MethodRegistry"),
    "EmbeddingRun": (".src.runs.contracts", "EmbeddingRun"),
    "FeatureSpec": (".src.runtime_contracts", "FeatureSpec"),
    "FitRequest": (".src.runtime_contracts", "FitRequest"),
    "InputSource": (".src.runtime_contracts", "InputSource"),
    "MethodConfig": (".src.runtime_contracts", "MethodConfig"),
    "RunCounts": (".src.runs.contracts", "RunCounts"),
    "WorkspaceConfig": (".src.workspaces", "WorkspaceConfig"),
    "builtin_workspaces_dir": (".src.workspaces", "builtin_workspaces_dir"),
    "default_method_registry": (".src.methods", "default_method_registry"),
    "list_runs": (".api", "list_runs"),
    "init_workspace": (".src.workspaces", "init_workspace"),
    "list_workspace_runs": (".api", "list_workspace_runs"),
    "load_workspace_config": (".src.workspaces", "load_workspace_config"),
    "register_method": (".src.methods", "register_method"),
    "registered_methods": (".src.methods", "registered_methods"),
    "run_analyze": (".api", "run_analyze"),
    "run_analyze_workspace": (".api", "run_analyze_workspace"),
    "run_fit": (".api", "run_fit"),
    "run_fit_workspace": (".api", "run_fit_workspace"),
    "run_sweep": (".api", "run_sweep"),
    "run_sweep_workspace": (".api", "run_sweep_workspace"),
    "run_umap": (".api", "run_umap"),
    "run_umap_workspace": (".api", "run_umap_workspace"),
    "supported_method_ids": (".src.methods", "supported_method_ids"),
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
