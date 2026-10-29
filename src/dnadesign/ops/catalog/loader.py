"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/catalog/loader.py

Catalog assembly for the shared runbook inventory.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from . import paths
from .metadata import (
    discover_catalog_metadata_paths,
    index_catalog_procedures,
    index_catalog_tool_routes,
    index_catalog_tool_sources,
    load_catalog_procedures,
    load_catalog_tool_sources,
    validate_related_tool_routes,
    validate_related_tools,
)
from .models import RunbookCatalog
from .packaged import packaged_catalog_path, read_catalog
from .paths import resolve_catalog_repo_root
from .provider_sources import discover_external_catalog_registry_sources


def load_runbook_catalog(*, repo_root: Path | None = None, include_external: bool = True) -> RunbookCatalog:
    resolved_repo_root = (
        resolve_catalog_repo_root(repo_root)
        if repo_root is not None
        else paths.discover_repo_root(Path.cwd()) or paths.discover_repo_root(Path(__file__))
    )
    external_sources = discover_external_catalog_registry_sources() if include_external else ()
    if resolved_repo_root is None:
        catalog_path = packaged_catalog_path()
        base_procedures, tool_sources, base_relations = read_catalog(catalog_path)
        procedures, procedure_relations = load_catalog_procedures(
            repo_root=catalog_path.parent,
            catalog_path=catalog_path,
            metadata_paths=(),
            external_sources=external_sources,
            base_procedures=base_procedures,
            base_relations=base_relations,
        )
    else:
        catalog_path = (resolved_repo_root / "docs" / "runbooks" / "README.md").resolve()
        if not catalog_path.exists():
            raise ValueError("runbook catalog missing: docs/runbooks/README.md")
        metadata_paths = discover_catalog_metadata_paths(resolved_repo_root)
        procedures, procedure_relations = load_catalog_procedures(
            repo_root=resolved_repo_root,
            catalog_path=catalog_path,
            metadata_paths=metadata_paths.registry_paths,
            external_sources=external_sources,
        )
        tool_sources = load_catalog_tool_sources(
            repo_root=resolved_repo_root,
            catalog_path=catalog_path,
            metadata_paths=metadata_paths.tool_source_paths,
        )
    procedure_index = index_catalog_procedures(procedures)
    tool_source_index = index_catalog_tool_sources(tool_sources)
    tool_route_index = index_catalog_tool_routes(tool_sources)
    validate_related_tools(procedure_index=procedure_index, tool_source_index=tool_source_index)
    validate_related_tool_routes(
        procedure_index=procedure_index,
        tool_source_index=tool_source_index,
        tool_route_index=tool_route_index,
    )
    return RunbookCatalog(
        repo_root=resolved_repo_root,
        catalog_path=catalog_path,
        procedures=procedures,
        tool_sources=tool_sources,
        procedure_index=procedure_index,
        procedure_relations=procedure_relations,
        tool_source_index=tool_source_index,
        tool_route_index=tool_route_index,
    )
