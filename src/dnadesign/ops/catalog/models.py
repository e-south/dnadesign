"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/catalog/models.py

Typed records for the Ops runbook catalog.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CatalogProcedureRelation:
    relation_type: str
    target_registry_id: str

    def as_dict(self) -> dict[str, str]:
        return {
            "type": self.relation_type,
            "target_registry_id": self.target_registry_id,
        }


@dataclass(frozen=True)
class CatalogProcedureToolRouteReference:
    tool: str
    route_id: str

    def as_dict(self) -> dict[str, str]:
        return {
            "tool": self.tool,
            "route_id": self.route_id,
        }


@dataclass(frozen=True)
class CatalogToolRouteEntry:
    tool: str
    route_id: str
    title: str
    doc_path: str
    summary: str

    def as_dict(self) -> dict[str, str]:
        return {
            "tool": self.tool,
            "route_id": self.route_id,
            "title": self.title,
            "doc_path": self.doc_path,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class CatalogProcedureEntry:
    registry_id: str
    title: str
    doc_path: str
    entry_type: str
    plane: str
    owner_boundary: str
    entry_artifact: str
    exit_artifact: str
    execution_kind: str
    status_kind: str
    summary: str
    catalog_order: int = field(repr=False)
    keywords: tuple[str, ...] = field(default_factory=tuple, repr=False)
    related_tools: tuple[str, ...] = field(default_factory=tuple, repr=False)
    related_tool_routes: tuple[CatalogProcedureToolRouteReference, ...] = field(default_factory=tuple, repr=False)


@dataclass(frozen=True)
class CatalogToolSourceEntry:
    tool: str
    title: str
    doc_path: str
    summary: str
    keywords: tuple[str, ...] = field(default_factory=tuple, repr=False)
    routes: tuple[CatalogToolRouteEntry, ...] = field(default_factory=tuple, repr=False)
    catalog_order: int = field(repr=False, default=0)


@dataclass(frozen=True)
class CatalogQuery:
    query: str | None = None
    entry_type: str | None = None
    plane: str | None = None
    execution_kind: str | None = None
    status_kind: str | None = None
    related_to: str | None = None
    tool: str | None = None

    def as_dict(self) -> dict[str, str]:
        filters: dict[str, str] = {}
        if self.query:
            filters["query"] = self.query
        if self.entry_type:
            filters["type"] = self.entry_type
        if self.plane:
            filters["plane"] = self.plane
        if self.execution_kind:
            filters["execution_kind"] = self.execution_kind
        if self.status_kind:
            filters["status_kind"] = self.status_kind
        if self.related_to:
            filters["related_to"] = self.related_to
        if self.tool:
            filters["tool"] = self.tool
        return filters

    def query_tokens(self) -> tuple[str, ...]:
        if not self.query:
            return ()
        return tuple(token for token in self.query.lower().split() if token)

    def has_procedure_filters(self) -> bool:
        return any((self.entry_type, self.plane, self.execution_kind, self.status_kind))

    def has_tool_filters(self) -> bool:
        return self.tool is not None


@dataclass(frozen=True)
class CatalogProcedureDetails:
    entry: CatalogProcedureEntry
    relations: tuple[CatalogProcedureRelation, ...]

    @property
    def owner_boundary(self) -> str:
        return self.entry.owner_boundary

    @property
    def entry_artifact(self) -> str:
        return self.entry.entry_artifact

    @property
    def exit_artifact(self) -> str:
        return self.entry.exit_artifact

    @property
    def related_registry_ids(self) -> tuple[str, ...]:
        return tuple(relation.target_registry_id for relation in self.relations)

    @property
    def related_tools(self) -> tuple[str, ...]:
        return self.entry.related_tools

    @property
    def related_tool_routes(self) -> tuple[CatalogProcedureToolRouteReference, ...]:
        return self.entry.related_tool_routes


@dataclass(frozen=True)
class RunbookCatalog:
    repo_root: Path
    catalog_path: Path
    procedures: tuple[CatalogProcedureEntry, ...]
    tool_sources: tuple[CatalogToolSourceEntry, ...]
    procedure_index: dict[str, CatalogProcedureEntry] = field(repr=False)
    procedure_relations: dict[str, tuple[CatalogProcedureRelation, ...]] = field(repr=False)
    tool_source_index: dict[str, CatalogToolSourceEntry] = field(repr=False)
    tool_route_index: dict[tuple[str, str], CatalogToolRouteEntry] = field(repr=False)

    def find_procedure(self, registry_id: str) -> CatalogProcedureEntry | None:
        return self.procedure_index.get(registry_id)

    def find_tool_source(self, tool: str) -> CatalogToolSourceEntry | None:
        return self.tool_source_index.get(tool)

    def find_tool_route(self, *, tool: str, route_id: str) -> CatalogToolRouteEntry | None:
        return self.tool_route_index.get((tool, route_id))
