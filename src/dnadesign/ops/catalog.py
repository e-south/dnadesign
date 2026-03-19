"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/catalog.py

Read-only catalog loader for the shared runbook inventory.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_TITLE_HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
_PROCEDURES_SECTION_HEADING = "### Authoritative cross-tool procedures"
_PROCEDURES_SECTION_INTRO = (
    "This table is generated from owner-local `*.registry.yaml` metadata sidecars. "
    "Edit those files instead of hand-editing rows here."
)
_TOOL_SOURCES_SECTION_HEADING = "### Tool-local runbook sources"
_TOOL_SOURCES_SECTION_INTRO = (
    "This table is generated from owner-local `*.tool-source.yaml` metadata sidecars. "
    "Edit those files instead of hand-editing rows here."
)
_REGISTRY_METADATA_SUFFIX = ".registry.yaml"
_TOOL_SOURCE_METADATA_SUFFIX = ".tool-source.yaml"
_ALLOWED_RELATION_TYPES = frozenset(
    {
        "alternative-to",
        "depends-on",
        "execution-support",
        "handoff-to",
        "see-also",
    }
)
_METADATA_TOKEN_PATTERN = re.compile(r"^[a-z][a-z0-9-]*(?:-[a-z0-9]+)*$")
_TOOL_SOURCE_KEYWORD_PATTERN = re.compile(r"^[a-z0-9][a-z0-9 _-]*$")


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
    progress_kind: str
    summary: str
    catalog_order: int = field(repr=False)
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
    progress_kind: str | None = None
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
        if self.progress_kind:
            filters["progress_kind"] = self.progress_kind
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
        return any((self.entry_type, self.plane, self.execution_kind, self.progress_kind))

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


def resolve_catalog_doc_path(*, catalog_path: Path, doc_path: str) -> Path:
    return (catalog_path.parent / doc_path).resolve()


def resolve_registry_metadata_path_for_doc_path(doc_path: Path | str) -> Path:
    normalized = Path(doc_path)
    return normalized.with_name(f"{normalized.stem}{_REGISTRY_METADATA_SUFFIX}")


def repo_relative_catalog_doc_path(*, repo_root: Path, catalog_path: Path, doc_path: str) -> str:
    resolved = resolve_catalog_doc_path(catalog_path=catalog_path, doc_path=doc_path)
    resolved_repo_root = repo_root.resolve()
    try:
        return str(resolved.relative_to(resolved_repo_root))
    except ValueError:
        return str(resolved)


def discover_repo_root(start: Path) -> Path | None:
    resolved = start.expanduser().resolve()
    anchor = resolved if resolved.is_dir() else resolved.parent
    for parent in (anchor, *anchor.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src" / "dnadesign").exists():
            return parent.resolve()
    return None


def resolve_catalog_repo_root(repo_root: Path | None) -> Path:
    if repo_root is not None:
        resolved = repo_root.expanduser().resolve()
        if not (resolved / "docs" / "runbooks" / "README.md").exists():
            raise ValueError("runbook catalog requires a repository checkout containing docs/runbooks/README.md")
        return resolved

    discovered = discover_repo_root(Path.cwd())
    if discovered is not None:
        return discovered

    discovered_from_module = discover_repo_root(Path(__file__))
    if discovered_from_module is not None:
        return discovered_from_module

    raise ValueError("runbook catalog requires a dnadesign repository checkout; pass --repo-root")


def load_runbook_catalog(*, repo_root: Path | None = None) -> RunbookCatalog:
    resolved_repo_root = resolve_catalog_repo_root(repo_root)
    catalog_path = (resolved_repo_root / "docs" / "runbooks" / "README.md").resolve()
    if not catalog_path.exists():
        raise ValueError("runbook catalog missing: docs/runbooks/README.md")

    procedures, procedure_relations = _load_catalog_procedures(
        repo_root=resolved_repo_root,
        catalog_path=catalog_path,
    )
    tool_sources = _load_catalog_tool_sources(
        repo_root=resolved_repo_root,
        catalog_path=catalog_path,
    )
    procedure_index = _index_catalog_procedures(procedures)
    tool_source_index = _index_catalog_tool_sources(tool_sources)
    tool_route_index = _index_catalog_tool_routes(tool_sources)
    _validate_related_tools(procedure_index=procedure_index, tool_source_index=tool_source_index)
    _validate_related_tool_routes(
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


def render_catalog_procedure_section(catalog: RunbookCatalog) -> str:
    lines = [
        _PROCEDURES_SECTION_INTRO,
        "",
        "| Registry id | Procedure | Type | Plane | Execution kind | Progress kind | Summary |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in catalog.procedures:
        lines.append(
            "| "
            f"`{entry.registry_id}` | "
            f"[{entry.title}]({entry.doc_path}) | "
            f"`{entry.entry_type}` | "
            f"`{entry.plane}` | "
            f"`{entry.execution_kind}` | "
            f"`{entry.progress_kind}` | "
            f"{entry.summary} |"
        )
    return "\n".join(lines)


def render_catalog_tool_source_section(catalog: RunbookCatalog) -> str:
    lines = [
        _TOOL_SOURCES_SECTION_INTRO,
        "",
        "| Tool | Docs entrypoint | What you will find |",
        "| --- | --- | --- |",
    ]
    for entry in catalog.tool_sources:
        lines.append(f"| `{entry.tool}` | [{entry.title}]({entry.doc_path}) | {entry.summary} |")
    return "\n".join(lines)


def rewrite_runbook_catalog_sections(*, repo_root: Path | None = None) -> Path:
    catalog = load_runbook_catalog(repo_root=repo_root)
    catalog_path = catalog.catalog_path
    catalog_text = catalog_path.read_text(encoding="utf-8")
    rendered_procedure_section = render_catalog_procedure_section(catalog)
    rendered_tool_source_section = render_catalog_tool_source_section(catalog)
    updated_text = _replace_markdown_section(
        text=catalog_text,
        heading=_PROCEDURES_SECTION_HEADING,
        body=rendered_procedure_section,
    )
    updated_text = _replace_markdown_section(
        text=updated_text,
        heading=_TOOL_SOURCES_SECTION_HEADING,
        body=rendered_tool_source_section,
    )
    catalog_path.write_text(updated_text.rstrip() + "\n", encoding="utf-8")
    return catalog_path


def rewrite_runbook_catalog_procedure_section(*, repo_root: Path | None = None) -> Path:
    return rewrite_runbook_catalog_sections(repo_root=repo_root)


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


def _load_catalog_procedures(
    *,
    repo_root: Path,
    catalog_path: Path,
) -> tuple[tuple[CatalogProcedureEntry, ...], dict[str, tuple[CatalogProcedureRelation, ...]]]:
    metadata_paths = tuple(sorted(_discover_registry_metadata_paths(repo_root)))
    if not metadata_paths:
        raise ValueError("runbook catalog requires at least one '*.registry.yaml' procedure metadata file")

    unsorted_entries: list[CatalogProcedureEntry] = []
    unsorted_relations: dict[str, tuple[CatalogProcedureRelation, ...]] = {}
    orders_by_value: dict[int, str] = {}
    for metadata_path in metadata_paths:
        entry, relations = _load_registry_metadata_file(
            metadata_path=metadata_path,
            repo_root=repo_root,
            catalog_path=catalog_path,
        )
        existing_registry_id = orders_by_value.get(entry.catalog_order)
        if existing_registry_id is not None:
            raise ValueError(
                "duplicate catalog_order in registry metadata: "
                f"{entry.catalog_order} used by both {existing_registry_id} and {entry.registry_id}"
            )
        orders_by_value[entry.catalog_order] = entry.registry_id
        unsorted_entries.append(entry)
        unsorted_relations[entry.registry_id] = relations

    indexed_entries = _index_catalog_procedures(tuple(unsorted_entries))
    for registry_id, relations in unsorted_relations.items():
        for relation in relations:
            if relation.target_registry_id not in indexed_entries:
                raise ValueError(f"registry relation target missing: {registry_id} -> {relation.target_registry_id}")
            if relation.target_registry_id == registry_id:
                raise ValueError(f"registry relation must not target itself: {registry_id}")

    procedures = tuple(sorted(unsorted_entries, key=lambda entry: (entry.catalog_order, entry.registry_id)))
    return procedures, unsorted_relations


def _load_catalog_tool_sources(
    *,
    repo_root: Path,
    catalog_path: Path,
) -> tuple[CatalogToolSourceEntry, ...]:
    metadata_paths = tuple(sorted(_discover_tool_source_metadata_paths(repo_root)))
    if not metadata_paths:
        return ()

    unsorted_entries: list[CatalogToolSourceEntry] = []
    orders_by_value: dict[int, str] = {}
    for metadata_path in metadata_paths:
        entry = _load_tool_source_metadata_file(
            metadata_path=metadata_path,
            repo_root=repo_root,
            catalog_path=catalog_path,
        )
        existing_tool = orders_by_value.get(entry.catalog_order)
        if existing_tool is not None:
            raise ValueError(
                "duplicate tool-source catalog_order in metadata: "
                f"{entry.catalog_order} used by both {existing_tool} and {entry.tool}"
            )
        orders_by_value[entry.catalog_order] = entry.tool
        unsorted_entries.append(entry)
    return tuple(sorted(unsorted_entries, key=lambda entry: (entry.catalog_order, entry.tool)))


def _discover_registry_metadata_paths(repo_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for search_root in _registry_metadata_search_roots(repo_root):
        for metadata_path in search_root.rglob(f"*{_REGISTRY_METADATA_SUFFIX}"):
            if any(segment in {"archived", "prototypes", "__pycache__"} for segment in metadata_path.parts):
                continue
            candidates.append(metadata_path.resolve())
    return candidates


def _discover_tool_source_metadata_paths(repo_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for search_root in _registry_metadata_search_roots(repo_root):
        for metadata_path in search_root.rglob(f"*{_TOOL_SOURCE_METADATA_SUFFIX}"):
            if any(segment in {"archived", "prototypes", "__pycache__"} for segment in metadata_path.parts):
                continue
            candidates.append(metadata_path.resolve())
    return candidates


def _registry_metadata_search_roots(repo_root: Path) -> tuple[Path, ...]:
    search_roots: list[Path] = []

    top_level_docs_root = (repo_root / "docs").resolve()
    if top_level_docs_root.exists():
        search_roots.append(top_level_docs_root)

    tool_src_root = (repo_root / "src" / "dnadesign").resolve()
    if tool_src_root.exists():
        for tool_root in sorted(path for path in tool_src_root.iterdir() if path.is_dir()):
            docs_root = (tool_root / "docs").resolve()
            if docs_root.exists():
                search_roots.append(docs_root)

    return tuple(search_roots)


def _load_registry_metadata_file(
    *,
    metadata_path: Path,
    repo_root: Path,
    catalog_path: Path,
) -> tuple[CatalogProcedureEntry, tuple[CatalogProcedureRelation, ...]]:
    payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"registry metadata must be a mapping: {metadata_path}")
    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"registry metadata schema_version must be 1: {metadata_path}")

    doc_path = _resolve_doc_path_for_metadata(metadata_path=metadata_path, repo_root=repo_root)
    title = _load_catalog_doc_title(doc_path)
    doc_link = _relative_catalog_doc_link(catalog_path=catalog_path, doc_path=doc_path)

    entry = CatalogProcedureEntry(
        registry_id=_required_string(payload, field_name="registry_id", metadata_path=metadata_path),
        title=title,
        doc_path=doc_link,
        entry_type=_required_string(payload, field_name="type", metadata_path=metadata_path),
        plane=_required_string(payload, field_name="plane", metadata_path=metadata_path),
        owner_boundary=_required_string(payload, field_name="owner_boundary", metadata_path=metadata_path),
        entry_artifact=_required_string(payload, field_name="entry_artifact", metadata_path=metadata_path),
        exit_artifact=_required_string(payload, field_name="exit_artifact", metadata_path=metadata_path),
        execution_kind=_required_string(payload, field_name="execution_kind", metadata_path=metadata_path),
        progress_kind=_required_string(payload, field_name="progress_kind", metadata_path=metadata_path),
        summary=_required_string(payload, field_name="summary", metadata_path=metadata_path),
        related_tools=_optional_related_tools(payload, metadata_path=metadata_path),
        related_tool_routes=_optional_related_tool_routes(payload, metadata_path=metadata_path),
        catalog_order=_required_positive_int(payload, field_name="catalog_order", metadata_path=metadata_path),
    )
    relations = _load_registry_relations(payload, metadata_path=metadata_path)
    return entry, relations


def _resolve_doc_path_for_metadata(*, metadata_path: Path, repo_root: Path) -> Path:
    return _resolve_doc_path_for_sidecar(
        metadata_path=metadata_path,
        repo_root=repo_root,
        suffix=_REGISTRY_METADATA_SUFFIX,
        error_prefix="registry metadata",
    )


def _load_tool_source_metadata_file(
    *,
    metadata_path: Path,
    repo_root: Path,
    catalog_path: Path,
) -> CatalogToolSourceEntry:
    payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"tool-source metadata must be a mapping: {metadata_path}")
    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"tool-source metadata schema_version must be 1: {metadata_path}")

    doc_path = _resolve_doc_path_for_sidecar(
        metadata_path=metadata_path,
        repo_root=repo_root,
        suffix=_TOOL_SOURCE_METADATA_SUFFIX,
        error_prefix="tool-source metadata",
    )
    title = _load_catalog_doc_title(doc_path)
    doc_link = _relative_catalog_doc_link(catalog_path=catalog_path, doc_path=doc_path)
    tool = _required_string(payload, field_name="tool", metadata_path=metadata_path)
    return CatalogToolSourceEntry(
        tool=tool,
        title=title,
        doc_path=doc_link,
        summary=_required_string(payload, field_name="summary", metadata_path=metadata_path),
        keywords=_optional_tool_source_keywords(payload, metadata_path=metadata_path),
        routes=_optional_tool_source_routes(
            payload,
            metadata_path=metadata_path,
            tool=tool,
            doc_root=doc_path.parent,
            catalog_path=catalog_path,
        ),
        catalog_order=_required_positive_int(payload, field_name="catalog_order", metadata_path=metadata_path),
    )


def _resolve_doc_path_for_sidecar(
    *,
    metadata_path: Path,
    repo_root: Path,
    suffix: str,
    error_prefix: str,
) -> Path:
    relative_metadata = metadata_path.resolve().relative_to(repo_root.resolve())
    if not relative_metadata.name.endswith(suffix):
        raise ValueError(f"invalid {error_prefix} filename: {metadata_path}")
    doc_relative = relative_metadata.with_name(relative_metadata.name[: -len(suffix)] + ".md")
    resolved_doc_path = (repo_root / doc_relative).resolve()
    if not resolved_doc_path.exists():
        raise ValueError(f"{error_prefix} doc missing: {resolved_doc_path}")
    return resolved_doc_path


def _relative_catalog_doc_link(*, catalog_path: Path, doc_path: Path) -> str:
    return os.path.relpath(doc_path, start=catalog_path.parent).replace(os.sep, "/")


def _load_catalog_doc_title(doc_path: Path) -> str:
    text = doc_path.read_text(encoding="utf-8")
    match = _TITLE_HEADING_PATTERN.search(text)
    if match is None:
        raise ValueError(f"{doc_path}: missing top-level markdown heading.")
    title = match.group(1).strip()
    if not title:
        raise ValueError(f"{doc_path}: top-level markdown heading must not be empty.")
    return title


def _required_string(payload: dict[str, object], *, field_name: str, metadata_path: Path) -> str:
    value = str(payload.get(field_name) or "").strip()
    if not value:
        raise ValueError(f"{metadata_path}: '{field_name}' must be a non-empty string.")
    if (
        field_name
        in {
            "type",
            "plane",
            "owner_boundary",
            "execution_kind",
            "progress_kind",
        }
        and _METADATA_TOKEN_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(f"{metadata_path}: '{field_name}' must use lowercase slug tokens.")
    return value


def _required_positive_int(payload: dict[str, object], *, field_name: str, metadata_path: Path) -> int:
    value = payload.get(field_name)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{metadata_path}: '{field_name}' must be a positive integer.")
    return value


def _load_registry_relations(
    payload: dict[str, object],
    *,
    metadata_path: Path,
) -> tuple[CatalogProcedureRelation, ...]:
    relations_payload = payload.get("relations") or []
    if not isinstance(relations_payload, list):
        raise ValueError(f"{metadata_path}: 'relations' must be a list when present.")
    relations: list[CatalogProcedureRelation] = []
    seen_targets: set[tuple[str, str]] = set()
    for index, relation_payload in enumerate(relations_payload, start=1):
        if not isinstance(relation_payload, dict):
            raise ValueError(f"{metadata_path}: relation {index} must be a mapping.")
        relation_type = str(relation_payload.get("type") or "").strip()
        target_registry_id = str(relation_payload.get("target") or "").strip()
        if not relation_type:
            raise ValueError(f"{metadata_path}: relation {index} missing 'type'.")
        if relation_type not in _ALLOWED_RELATION_TYPES:
            allowed = ", ".join(sorted(_ALLOWED_RELATION_TYPES))
            raise ValueError(f"{metadata_path}: relation {index} type must be one of: {allowed}.")
        if not target_registry_id:
            raise ValueError(f"{metadata_path}: relation {index} missing 'target'.")
        relation_key = (relation_type, target_registry_id)
        if relation_key in seen_targets:
            raise ValueError(f"{metadata_path}: duplicate relation '{relation_type}' -> '{target_registry_id}'.")
        seen_targets.add(relation_key)
        relations.append(
            CatalogProcedureRelation(
                relation_type=relation_type,
                target_registry_id=target_registry_id,
            )
        )
    return tuple(relations)


def _optional_related_tools(
    payload: dict[str, object],
    *,
    metadata_path: Path,
) -> tuple[str, ...]:
    related_tools_payload = payload.get("related_tools") or []
    if not isinstance(related_tools_payload, list):
        raise ValueError(f"{metadata_path}: 'related_tools' must be a list when present.")
    related_tools: list[str] = []
    seen: set[str] = set()
    for index, tool_payload in enumerate(related_tools_payload, start=1):
        tool = str(tool_payload or "").strip()
        if not tool:
            raise ValueError(f"{metadata_path}: related_tools[{index}] must be a non-empty string.")
        if _METADATA_TOKEN_PATTERN.fullmatch(tool) is None:
            raise ValueError(f"{metadata_path}: related_tools[{index}] must use lowercase slug tokens.")
        if tool in seen:
            raise ValueError(f"{metadata_path}: duplicate related tool '{tool}'.")
        seen.add(tool)
        related_tools.append(tool)
    return tuple(related_tools)


def _optional_related_tool_routes(
    payload: dict[str, object],
    *,
    metadata_path: Path,
) -> tuple[CatalogProcedureToolRouteReference, ...]:
    route_refs_payload = payload.get("related_tool_routes") or []
    if not isinstance(route_refs_payload, list):
        raise ValueError(f"{metadata_path}: 'related_tool_routes' must be a list when present.")
    route_refs: list[CatalogProcedureToolRouteReference] = []
    seen: set[tuple[str, str]] = set()
    for index, route_payload in enumerate(route_refs_payload, start=1):
        if not isinstance(route_payload, dict):
            raise ValueError(f"{metadata_path}: related_tool_routes[{index}] must be a mapping.")
        tool = str(route_payload.get("tool") or "").strip()
        route_id = str(route_payload.get("route") or "").strip()
        if not tool:
            raise ValueError(f"{metadata_path}: related_tool_routes[{index}] missing 'tool'.")
        if _METADATA_TOKEN_PATTERN.fullmatch(tool) is None:
            raise ValueError(f"{metadata_path}: related_tool_routes[{index}] tool must use lowercase slug tokens.")
        if not route_id:
            raise ValueError(f"{metadata_path}: related_tool_routes[{index}] missing 'route'.")
        if _METADATA_TOKEN_PATTERN.fullmatch(route_id) is None:
            raise ValueError(f"{metadata_path}: related_tool_routes[{index}] route must use lowercase slug tokens.")
        route_key = (tool, route_id)
        if route_key in seen:
            raise ValueError(f"{metadata_path}: duplicate related tool route '{tool}/{route_id}'.")
        seen.add(route_key)
        route_refs.append(CatalogProcedureToolRouteReference(tool=tool, route_id=route_id))
    return tuple(route_refs)


def _optional_tool_source_keywords(
    payload: dict[str, object],
    *,
    metadata_path: Path,
) -> tuple[str, ...]:
    keywords_payload = payload.get("keywords") or []
    if not isinstance(keywords_payload, list):
        raise ValueError(f"{metadata_path}: 'keywords' must be a list when present.")
    keywords: list[str] = []
    seen: set[str] = set()
    for index, keyword_payload in enumerate(keywords_payload, start=1):
        keyword = str(keyword_payload or "").strip().lower()
        if not keyword:
            raise ValueError(f"{metadata_path}: keyword {index} must be a non-empty string.")
        if _TOOL_SOURCE_KEYWORD_PATTERN.fullmatch(keyword) is None:
            raise ValueError(
                f"{metadata_path}: keyword {index} must use lowercase words, spaces, hyphens, or underscores."
            )
        if keyword in seen:
            raise ValueError(f"{metadata_path}: duplicate keyword '{keyword}'.")
        seen.add(keyword)
        keywords.append(keyword)
    return tuple(keywords)


def _optional_tool_source_routes(
    payload: dict[str, object],
    *,
    metadata_path: Path,
    tool: str,
    doc_root: Path,
    catalog_path: Path,
) -> tuple[CatalogToolRouteEntry, ...]:
    routes_payload = payload.get("routes") or []
    if not isinstance(routes_payload, list):
        raise ValueError(f"{metadata_path}: 'routes' must be a list when present.")
    routes: list[CatalogToolRouteEntry] = []
    seen: set[str] = set()
    docs_root = doc_root.resolve()
    for index, route_payload in enumerate(routes_payload, start=1):
        if not isinstance(route_payload, dict):
            raise ValueError(f"{metadata_path}: route {index} must be a mapping.")
        route_id = str(route_payload.get("id") or "").strip()
        relative_path = str(route_payload.get("path") or "").strip()
        summary = str(route_payload.get("summary") or "").strip()
        if not route_id:
            raise ValueError(f"{metadata_path}: route {index} missing 'id'.")
        if _METADATA_TOKEN_PATTERN.fullmatch(route_id) is None:
            raise ValueError(f"{metadata_path}: route {index} id must use lowercase slug tokens.")
        if route_id in seen:
            raise ValueError(f"{metadata_path}: duplicate route id '{route_id}'.")
        if not relative_path:
            raise ValueError(f"{metadata_path}: route {index} missing 'path'.")
        if Path(relative_path).is_absolute():
            raise ValueError(f"{metadata_path}: route {index} path must be relative to the tool docs root.")
        if not summary:
            raise ValueError(f"{metadata_path}: route {index} missing 'summary'.")
        resolved_doc_path = (docs_root / relative_path).resolve()
        try:
            resolved_doc_path.relative_to(docs_root)
        except ValueError as exc:
            raise ValueError(
                f"{metadata_path}: route {index} path must stay inside the tool docs root: {relative_path}"
            ) from exc
        if not resolved_doc_path.exists():
            raise ValueError(f"{metadata_path}: route {index} doc missing: {resolved_doc_path}")
        seen.add(route_id)
        routes.append(
            CatalogToolRouteEntry(
                tool=tool,
                route_id=route_id,
                title=_load_catalog_doc_title(resolved_doc_path),
                doc_path=_relative_catalog_doc_link(catalog_path=catalog_path, doc_path=resolved_doc_path),
                summary=summary,
            )
        )
    return tuple(routes)


def _extract_table_rows(*, text: str, heading: str) -> list[list[str]]:
    lines = text.splitlines()
    try:
        heading_index = next(index for index, line in enumerate(lines) if line.strip() == heading)
    except StopIteration as exc:
        raise ValueError(f"runbook catalog missing section: {heading}") from exc

    table_lines: list[str] = []
    for line in lines[heading_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            if table_lines:
                break
            continue
        if stripped.startswith("### "):
            break
        if stripped.startswith("|"):
            table_lines.append(stripped)
            continue
        if table_lines:
            break

    if len(table_lines) < 3:
        raise ValueError(f"runbook catalog section has no data table: {heading}")

    rows = [_split_markdown_row(line) for line in table_lines]
    return rows[2:]


def _replace_markdown_section(*, text: str, heading: str, body: str) -> str:
    lines = text.splitlines()
    try:
        heading_index = next(index for index, line in enumerate(lines) if line.strip() == heading)
    except StopIteration as exc:
        raise ValueError(f"runbook catalog missing section: {heading}") from exc

    end_index = len(lines)
    for index in range(heading_index + 1, len(lines)):
        if lines[index].strip().startswith("### "):
            end_index = index
            break
    replacement = [lines[heading_index], "", *body.rstrip().splitlines()]
    if end_index < len(lines):
        replacement.append("")
    updated_lines = [*lines[:heading_index], *replacement, *lines[end_index:]]
    return "\n".join(updated_lines)


def _split_markdown_row(row: str) -> list[str]:
    stripped = row.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        raise ValueError(f"invalid markdown table row: {row}")
    return [cell.strip() for cell in stripped[1:-1].split("|")]


def _parse_link_cell(cell: str) -> tuple[str, str]:
    match = _LINK_PATTERN.search(cell)
    if match is None:
        raise ValueError(f"expected markdown link cell, got: {cell}")
    return match.group(1).strip(), match.group(2).strip()


def _strip_ticks(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("`") and stripped.endswith("`") and len(stripped) >= 2:
        return stripped[1:-1]
    return stripped


def _parse_tool_source_row(row: list[str]) -> CatalogToolSourceEntry:
    if len(row) != 3:
        raise ValueError(f"expected 3 columns for tool-source row, got {len(row)}: {row}")
    title, doc_path = _parse_link_cell(row[1])
    return CatalogToolSourceEntry(
        tool=_strip_ticks(row[0]),
        title=title,
        doc_path=doc_path,
        summary=row[2].strip(),
    )


def _index_catalog_procedures(
    procedures: tuple[CatalogProcedureEntry, ...],
) -> dict[str, CatalogProcedureEntry]:
    indexed: dict[str, CatalogProcedureEntry] = {}
    for entry in procedures:
        if entry.registry_id in indexed:
            raise ValueError(f"duplicate registry id in runbook catalog: {entry.registry_id}")
        indexed[entry.registry_id] = entry
    return indexed


def _index_catalog_tool_sources(
    tool_sources: tuple[CatalogToolSourceEntry, ...],
) -> dict[str, CatalogToolSourceEntry]:
    indexed: dict[str, CatalogToolSourceEntry] = {}
    for entry in tool_sources:
        if entry.tool in indexed:
            raise ValueError(f"duplicate tool source in runbook catalog: {entry.tool}")
        indexed[entry.tool] = entry
    return indexed


def _index_catalog_tool_routes(
    tool_sources: tuple[CatalogToolSourceEntry, ...],
) -> dict[tuple[str, str], CatalogToolRouteEntry]:
    indexed: dict[tuple[str, str], CatalogToolRouteEntry] = {}
    for entry in tool_sources:
        for route in entry.routes:
            key = (route.tool, route.route_id)
            if key in indexed:
                raise ValueError(f"duplicate tool route in runbook catalog: {route.tool}/{route.route_id}")
            indexed[key] = route
    return indexed


def _validate_related_tools(
    *,
    procedure_index: dict[str, CatalogProcedureEntry],
    tool_source_index: dict[str, CatalogToolSourceEntry],
) -> None:
    for entry in procedure_index.values():
        for tool in entry.related_tools:
            if tool == entry.owner_boundary:
                raise ValueError(
                    "registry related tool duplicates owner boundary: "
                    f"{entry.registry_id} -> {tool}. Owner docs are already surfaced separately."
                )
            if tool not in tool_source_index:
                raise ValueError(f"registry related tool missing from catalog: {entry.registry_id} -> {tool}")


def _validate_related_tool_routes(
    *,
    procedure_index: dict[str, CatalogProcedureEntry],
    tool_source_index: dict[str, CatalogToolSourceEntry],
    tool_route_index: dict[tuple[str, str], CatalogToolRouteEntry],
) -> None:
    for entry in procedure_index.values():
        for reference in entry.related_tool_routes:
            if reference.tool == entry.owner_boundary:
                raise ValueError(
                    "registry related tool route duplicates owner boundary: "
                    f"{entry.registry_id} -> {reference.tool}/{reference.route_id}"
                )
            if reference.tool not in tool_source_index:
                raise ValueError(
                    "registry related tool route references unknown tool: "
                    f"{entry.registry_id} -> {reference.tool}/{reference.route_id}"
                )
            if reference.tool not in entry.related_tools:
                raise ValueError(
                    "registry related tool route requires matching related_tools entry: "
                    f"{entry.registry_id} -> {reference.tool}/{reference.route_id}"
                )
            if (reference.tool, reference.route_id) not in tool_route_index:
                raise ValueError(
                    "registry related tool route missing from catalog: "
                    f"{entry.registry_id} -> {reference.tool}/{reference.route_id}"
                )


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
    if query.progress_kind is not None and entry.progress_kind != query.progress_kind:
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
            entry.progress_kind,
            entry.summary,
            entry.entry_artifact,
            entry.exit_artifact,
        )
    ).lower()


def _tool_source_haystack(entry: CatalogToolSourceEntry) -> str:
    return " ".join((entry.tool, entry.title, entry.doc_path, entry.summary, *entry.keywords)).lower()
