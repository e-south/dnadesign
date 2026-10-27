"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/catalog/__init__.py

Public catalog surface for the shared runbook inventory.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .loader import load_runbook_catalog
from .models import (
    CatalogProcedureDetails,
    CatalogProcedureEntry,
    CatalogProcedureRelation,
    CatalogProcedureToolRouteReference,
    CatalogQuery,
    CatalogToolRouteEntry,
    CatalogToolSourceEntry,
    RunbookCatalog,
)
from .paths import (
    discover_repo_root,
    repo_relative_catalog_doc_path,
    resolve_catalog_doc_path,
    resolve_catalog_repo_root,
    resolve_registry_metadata_path_for_doc_path,
)
from .query import (
    filter_runbook_catalog,
    load_catalog_procedure_details,
    load_catalog_procedure_owner_boundary,
    load_catalog_related_registry_ids,
    load_catalog_related_tool_routes,
    load_catalog_related_tool_sources,
    resolve_catalog_procedure_entry,
    suggest_procedure_registry_ids,
)
from .rendering import (
    render_catalog_procedure_section,
    render_catalog_tool_source_section,
    rewrite_runbook_catalog_procedure_section,
    rewrite_runbook_catalog_sections,
)

__all__ = [
    "CatalogProcedureDetails",
    "CatalogProcedureEntry",
    "CatalogProcedureRelation",
    "CatalogProcedureToolRouteReference",
    "CatalogQuery",
    "CatalogToolRouteEntry",
    "CatalogToolSourceEntry",
    "RunbookCatalog",
    "discover_repo_root",
    "filter_runbook_catalog",
    "load_catalog_procedure_details",
    "load_catalog_procedure_owner_boundary",
    "load_catalog_related_registry_ids",
    "load_catalog_related_tool_routes",
    "load_catalog_related_tool_sources",
    "load_runbook_catalog",
    "render_catalog_procedure_section",
    "render_catalog_tool_source_section",
    "repo_relative_catalog_doc_path",
    "resolve_catalog_doc_path",
    "resolve_catalog_procedure_entry",
    "resolve_catalog_repo_root",
    "resolve_registry_metadata_path_for_doc_path",
    "rewrite_runbook_catalog_procedure_section",
    "rewrite_runbook_catalog_sections",
    "suggest_procedure_registry_ids",
]
