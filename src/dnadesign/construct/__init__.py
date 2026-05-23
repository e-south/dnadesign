"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/__init__.py

Public construct package exports.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .src.cli import main
    from .src.interfaces.api import (
        JobConfig,
        LinearSsdnaCompositionResult,
        LinearSsdnaCompositionSummary,
        PreflightResult,
        RunResult,
        load_job_config,
        load_linear_ssdna_composition_config,
        preflight_from_config,
        publish_composition_review_svg,
        run_from_config,
        run_linear_ssdna_composition,
        summarize_linear_ssdna_composition,
    )
    from .src.interfaces.contracts import (
        ConstructUSROutputContract,
        list_construct_workspace_selectors,
        list_construct_workspace_selectors_from_root,
        list_construct_workspaces,
        list_construct_workspaces_from_root,
        resolve_construct_run_id_from_config,
        resolve_construct_usr_output_contract,
        resolve_construct_workspace_config_path_from_root,
        resolve_construct_workspace_project_id_from_config,
        resolve_construct_workspace_root_from_config,
    )

__all__ = [
    "ConstructUSROutputContract",
    "JobConfig",
    "LinearSsdnaCompositionResult",
    "LinearSsdnaCompositionSummary",
    "PreflightResult",
    "RunResult",
    "list_construct_workspace_selectors",
    "list_construct_workspace_selectors_from_root",
    "list_construct_workspaces",
    "list_construct_workspaces_from_root",
    "load_job_config",
    "load_linear_ssdna_composition_config",
    "main",
    "preflight_from_config",
    "publish_composition_review_svg",
    "resolve_construct_run_id_from_config",
    "resolve_construct_usr_output_contract",
    "resolve_construct_workspace_config_path_from_root",
    "resolve_construct_workspace_project_id_from_config",
    "resolve_construct_workspace_root_from_config",
    "run_from_config",
    "run_linear_ssdna_composition",
    "summarize_linear_ssdna_composition",
]

_EXPORT_MODULES = {
    "ConstructUSROutputContract": ".src.interfaces.contracts",
    "JobConfig": ".src.interfaces.api",
    "LinearSsdnaCompositionResult": ".src.interfaces.api",
    "LinearSsdnaCompositionSummary": ".src.interfaces.api",
    "PreflightResult": ".src.interfaces.api",
    "RunResult": ".src.interfaces.api",
    "list_construct_workspace_selectors": ".src.interfaces.contracts",
    "list_construct_workspace_selectors_from_root": ".src.interfaces.contracts",
    "list_construct_workspaces": ".src.interfaces.contracts",
    "list_construct_workspaces_from_root": ".src.interfaces.contracts",
    "load_job_config": ".src.interfaces.api",
    "load_linear_ssdna_composition_config": ".src.interfaces.api",
    "main": ".src.cli",
    "preflight_from_config": ".src.interfaces.api",
    "publish_composition_review_svg": ".src.interfaces.api",
    "resolve_construct_run_id_from_config": ".src.interfaces.contracts",
    "resolve_construct_usr_output_contract": ".src.interfaces.contracts",
    "resolve_construct_workspace_config_path_from_root": ".src.interfaces.contracts",
    "resolve_construct_workspace_project_id_from_config": ".src.interfaces.contracts",
    "resolve_construct_workspace_root_from_config": ".src.interfaces.contracts",
    "run_from_config": ".src.interfaces.api",
    "run_linear_ssdna_composition": ".src.interfaces.api",
    "summarize_linear_ssdna_composition": ".src.interfaces.api",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
