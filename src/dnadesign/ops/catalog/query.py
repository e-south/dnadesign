"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/catalog/query.py

Query helpers for the Ops runbook catalog.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import difflib

from .models import (
    CatalogProcedureDetails,
    CatalogProcedureEntry,
    CatalogQuery,
    CatalogToolRouteEntry,
    CatalogToolSourceEntry,
    RunbookCatalog,
)


def filter_runbook_catalog(
    catalog: RunbookCatalog,
    *,
    query: CatalogQuery,
) -> tuple[tuple[CatalogProcedureEntry, ...], tuple[CatalogToolSourceEntry, ...]]:
    query_tokens = query.query_tokens()
    related_registry_ids: frozenset[str] | None = None
    related_tool_ids: frozenset[str] | None = None
    if query.related_to is not None:
        related_registry_ids = frozenset(load_catalog_related_registry_ids(catalog, query.related_to))
        related_tool_ids = frozenset(
            entry.tool for entry in load_catalog_related_tool_sources(catalog, query.related_to)
        )
    procedures = tuple(
        entry
        for entry in catalog.procedures
        if _matches_procedure_query(
            entry=entry,
            query=query,
            query_tokens=query_tokens,
            related_registry_ids=related_registry_ids,
        )
    )
    tool_sources = tuple(
        entry
        for entry in catalog.tool_sources
        if _matches_tool_source_query(
            entry=entry,
            query=query,
            query_tokens=query_tokens,
            related_tool_ids=related_tool_ids,
        )
    )
    return procedures, tool_sources


def suggest_procedure_registry_ids(
    catalog: RunbookCatalog,
    registry_id: str,
    *,
    limit: int = 3,
) -> tuple[str, ...]:
    available_ids = tuple(entry.registry_id for entry in catalog.procedures)
    normalized = registry_id.strip().lower()
    if not normalized:
        return ()

    prefix_matches = [candidate for candidate in available_ids if candidate.lower().startswith(normalized)]
    substring_matches = [
        candidate for candidate in available_ids if normalized in candidate.lower() and candidate not in prefix_matches
    ]
    fuzzy_matches = [
        candidate
        for candidate in difflib.get_close_matches(registry_id, available_ids, n=limit, cutoff=0.35)
        if candidate not in prefix_matches and candidate not in substring_matches
    ]
    ordered = tuple((*prefix_matches, *substring_matches, *fuzzy_matches))
    return ordered[:limit]


def load_catalog_procedure_details(
    catalog: RunbookCatalog,
    procedure: CatalogProcedureEntry | str,
) -> CatalogProcedureDetails:
    entry = resolve_catalog_procedure_entry(catalog, procedure)
    return CatalogProcedureDetails(
        entry=entry,
        relations=catalog.procedure_relations.get(entry.registry_id, ()),
    )


def load_catalog_related_registry_ids(
    catalog: RunbookCatalog,
    procedure: CatalogProcedureEntry | str,
    *,
    include_self: bool = False,
) -> tuple[str, ...]:
    details = load_catalog_procedure_details(catalog, procedure)
    if include_self:
        return (details.entry.registry_id, *details.related_registry_ids)
    return details.related_registry_ids


def load_catalog_related_tool_sources(
    catalog: RunbookCatalog,
    procedure: CatalogProcedureEntry | str,
) -> tuple[CatalogToolSourceEntry, ...]:
    details = load_catalog_procedure_details(catalog, procedure)
    related_sources: list[CatalogToolSourceEntry] = []
    for tool in details.related_tools:
        entry = catalog.find_tool_source(tool)
        if entry is None:
            raise ValueError(f"registry related tool missing from catalog: {details.entry.registry_id} -> {tool}")
        related_sources.append(entry)
    return tuple(related_sources)


def load_catalog_related_tool_routes(
    catalog: RunbookCatalog,
    procedure: CatalogProcedureEntry | str,
) -> tuple[CatalogToolRouteEntry, ...]:
    details = load_catalog_procedure_details(catalog, procedure)
    related_routes: list[CatalogToolRouteEntry] = []
    for reference in details.related_tool_routes:
        entry = catalog.find_tool_route(tool=reference.tool, route_id=reference.route_id)
        if entry is None:
            raise ValueError(
                "registry related tool route missing from catalog: "
                f"{details.entry.registry_id} -> {reference.tool}/{reference.route_id}"
            )
        related_routes.append(entry)
    return tuple(related_routes)


def load_catalog_procedure_owner_boundary(
    catalog: RunbookCatalog,
    procedure: CatalogProcedureEntry | str,
) -> str:
    entry = resolve_catalog_procedure_entry(catalog, procedure)
    return entry.owner_boundary


def resolve_catalog_procedure_entry(
    catalog: RunbookCatalog,
    procedure: CatalogProcedureEntry | str,
) -> CatalogProcedureEntry:
    if isinstance(procedure, CatalogProcedureEntry):
        return procedure
    entry = catalog.find_procedure(procedure)
    if entry is None:
        raise ValueError(f"unknown registry id: {procedure}")
    return entry


def _matches_procedure_query(
    *,
    entry: CatalogProcedureEntry,
    query: CatalogQuery,
    query_tokens: tuple[str, ...],
    related_registry_ids: frozenset[str] | None,
) -> bool:
    if query.has_tool_filters():
        return False
    if query.entry_type is not None and entry.entry_type != query.entry_type:
        return False
    if query.plane is not None and entry.plane != query.plane:
        return False
    if query.execution_kind is not None and entry.execution_kind != query.execution_kind:
        return False
    if query.status_kind is not None and entry.status_kind != query.status_kind:
        return False
    if related_registry_ids is not None and entry.registry_id not in related_registry_ids:
        return False
    if query_tokens and not _query_tokens_match(_procedure_haystack(entry), query_tokens):
        return False
    return True


def _matches_tool_source_query(
    *,
    entry: CatalogToolSourceEntry,
    query: CatalogQuery,
    query_tokens: tuple[str, ...],
    related_tool_ids: frozenset[str] | None,
) -> bool:
    if query.has_procedure_filters():
        return False
    if query.tool is not None and entry.tool != query.tool:
        return False
    if related_tool_ids is not None and entry.tool not in related_tool_ids:
        return False
    if query_tokens and not _query_tokens_match(_tool_source_haystack(entry), query_tokens):
        return False
    return True


def _query_tokens_match(haystack: str, query_tokens: tuple[str, ...]) -> bool:
    return all(token in haystack for token in query_tokens)


def _procedure_haystack(entry: CatalogProcedureEntry) -> str:
    return " ".join(
        (
            entry.registry_id,
            entry.title,
            entry.doc_path,
            entry.entry_type,
            entry.plane,
            entry.owner_boundary,
            entry.execution_kind,
            entry.status_kind,
            entry.summary,
            entry.entry_artifact,
            entry.exit_artifact,
            *entry.keywords,
        )
    ).lower()


def _tool_source_haystack(entry: CatalogToolSourceEntry) -> str:
    return " ".join((entry.tool, entry.title, entry.doc_path, entry.summary, *entry.keywords)).lower()
