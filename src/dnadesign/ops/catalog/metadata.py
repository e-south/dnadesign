"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/catalog/metadata.py

Sidecar metadata loading and validation for the Ops runbook catalog.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from dnadesign.ops.discovery import discover_suffixed_files

from .constants import (
    ALLOWED_RELATION_TYPES,
    METADATA_TOKEN_PATTERN,
    REGISTRY_METADATA_SUFFIX,
    TOOL_SOURCE_KEYWORD_PATTERN,
    TOOL_SOURCE_METADATA_SUFFIX,
)
from .models import (
    CatalogProcedureEntry,
    CatalogProcedureRelation,
    CatalogProcedureToolRouteReference,
    CatalogToolRouteEntry,
    CatalogToolSourceEntry,
)
from .paths import (
    catalog_metadata_search_roots,
    load_catalog_doc_title,
    relative_catalog_doc_link,
    resolve_doc_path_for_metadata,
    resolve_doc_path_for_sidecar,
)
from .provider_sources import CatalogRegistrySource

_REGISTRY_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "catalog_order",
        "registry_id",
        "type",
        "plane",
        "owner_boundary",
        "entry_artifact",
        "exit_artifact",
        "execution_kind",
        "status_kind",
        "summary",
        "keywords",
        "related_tools",
        "related_tool_routes",
        "relations",
    }
)
_TOOL_SOURCE_METADATA_KEYS = frozenset({"schema_version", "catalog_order", "tool", "summary", "keywords", "routes"})
_REGISTRY_RELATION_KEYS = frozenset({"type", "target"})
_RELATED_TOOL_ROUTE_KEYS = frozenset({"tool", "route"})
_TOOL_SOURCE_ROUTE_KEYS = frozenset({"id", "path", "summary"})


@dataclass(frozen=True)
class CatalogMetadataPaths:
    registry_paths: tuple[Path, ...]
    tool_source_paths: tuple[Path, ...]


def discover_catalog_metadata_paths(repo_root: Path) -> CatalogMetadataPaths:
    """Discover both catalog sidecar kinds in one bounded traversal."""

    candidates = discover_suffixed_files(
        roots=catalog_metadata_search_roots(repo_root),
        suffixes=(REGISTRY_METADATA_SUFFIX, TOOL_SOURCE_METADATA_SUFFIX),
    )
    return CatalogMetadataPaths(
        registry_paths=tuple(path for path in candidates if path.name.endswith(REGISTRY_METADATA_SUFFIX)),
        tool_source_paths=tuple(path for path in candidates if path.name.endswith(TOOL_SOURCE_METADATA_SUFFIX)),
    )


def load_catalog_procedures(
    *,
    repo_root: Path,
    catalog_path: Path,
    metadata_paths: tuple[Path, ...] | None = None,
    external_sources: tuple[CatalogRegistrySource, ...] = (),
) -> tuple[tuple[CatalogProcedureEntry, ...], dict[str, tuple[CatalogProcedureRelation, ...]]]:
    if metadata_paths is None:
        metadata_paths = discover_catalog_metadata_paths(repo_root).registry_paths
    if not metadata_paths and not external_sources:
        raise ValueError("runbook catalog requires at least one '*.registry.yaml' procedure metadata file")

    unsorted_entries: list[CatalogProcedureEntry] = []
    unsorted_relations: dict[str, tuple[CatalogProcedureRelation, ...]] = {}
    orders_by_provider: dict[Path, dict[int, str]] = {}
    sources = (
        *(CatalogRegistrySource(path=path, package_root=repo_root) for path in metadata_paths),
        *external_sources,
    )
    for source in sources:
        provider_root = source.package_root.expanduser().resolve()
        orders_by_value = orders_by_provider.setdefault(provider_root, {})
        entry, relations = _load_registry_metadata_file(
            metadata_path=source.path,
            repo_root=source.package_root,
            catalog_path=catalog_path,
        )
        existing_registry_id = orders_by_value.get(entry.catalog_order)
        if existing_registry_id is not None:
            raise ValueError(
                "duplicate catalog_order within one registry provider: "
                f"{entry.catalog_order} used by both {existing_registry_id} and {entry.registry_id} "
                f"under {provider_root}"
            )
        orders_by_value[entry.catalog_order] = entry.registry_id
        unsorted_entries.append(entry)
        unsorted_relations[entry.registry_id] = relations

    indexed_entries = index_catalog_procedures(tuple(unsorted_entries))
    for registry_id, relations in unsorted_relations.items():
        for relation in relations:
            if relation.target_registry_id not in indexed_entries:
                raise ValueError(f"registry relation target missing: {registry_id} -> {relation.target_registry_id}")
            if relation.target_registry_id == registry_id:
                raise ValueError(f"registry relation must not target itself: {registry_id}")

    procedures = tuple(sorted(unsorted_entries, key=lambda entry: (entry.catalog_order, entry.registry_id)))
    return procedures, unsorted_relations


def load_catalog_tool_sources(
    *,
    repo_root: Path,
    catalog_path: Path,
    metadata_paths: tuple[Path, ...] | None = None,
) -> tuple[CatalogToolSourceEntry, ...]:
    if metadata_paths is None:
        metadata_paths = discover_catalog_metadata_paths(repo_root).tool_source_paths
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


def index_catalog_procedures(
    procedures: tuple[CatalogProcedureEntry, ...],
) -> dict[str, CatalogProcedureEntry]:
    indexed: dict[str, CatalogProcedureEntry] = {}
    for entry in procedures:
        if entry.registry_id in indexed:
            raise ValueError(f"duplicate registry id in runbook catalog: {entry.registry_id}")
        indexed[entry.registry_id] = entry
    return indexed


def index_catalog_tool_sources(
    tool_sources: tuple[CatalogToolSourceEntry, ...],
) -> dict[str, CatalogToolSourceEntry]:
    indexed: dict[str, CatalogToolSourceEntry] = {}
    for entry in tool_sources:
        if entry.tool in indexed:
            raise ValueError(f"duplicate tool source in runbook catalog: {entry.tool}")
        indexed[entry.tool] = entry
    return indexed


def index_catalog_tool_routes(
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


def validate_related_tools(
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


def validate_related_tool_routes(
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


def _load_registry_metadata_file(
    *,
    metadata_path: Path,
    repo_root: Path,
    catalog_path: Path,
) -> tuple[CatalogProcedureEntry, tuple[CatalogProcedureRelation, ...]]:
    payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"registry metadata must be a mapping: {metadata_path}")
    _reject_unknown_keys(
        payload,
        allowed_keys=_REGISTRY_METADATA_KEYS,
        label="registry metadata",
        metadata_path=metadata_path,
    )
    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"registry metadata schema_version must be 1: {metadata_path}")

    doc_path = resolve_doc_path_for_metadata(metadata_path=metadata_path, repo_root=repo_root)
    title = load_catalog_doc_title(doc_path)
    doc_link = relative_catalog_doc_link(catalog_path=catalog_path, doc_path=doc_path)

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
        status_kind=_required_string(payload, field_name="status_kind", metadata_path=metadata_path),
        summary=_required_string(payload, field_name="summary", metadata_path=metadata_path),
        keywords=_optional_metadata_keywords(payload, metadata_path=metadata_path),
        related_tools=_optional_related_tools(payload, metadata_path=metadata_path),
        related_tool_routes=_optional_related_tool_routes(payload, metadata_path=metadata_path),
        catalog_order=_required_positive_int(payload, field_name="catalog_order", metadata_path=metadata_path),
    )
    relations = _load_registry_relations(payload, metadata_path=metadata_path)
    return entry, relations


def _load_tool_source_metadata_file(
    *,
    metadata_path: Path,
    repo_root: Path,
    catalog_path: Path,
) -> CatalogToolSourceEntry:
    payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"tool-source metadata must be a mapping: {metadata_path}")
    _reject_unknown_keys(
        payload,
        allowed_keys=_TOOL_SOURCE_METADATA_KEYS,
        label="tool-source metadata",
        metadata_path=metadata_path,
    )
    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"tool-source metadata schema_version must be 1: {metadata_path}")

    doc_path = resolve_doc_path_for_sidecar(
        metadata_path=metadata_path,
        repo_root=repo_root,
        suffix=TOOL_SOURCE_METADATA_SUFFIX,
        error_prefix="tool-source metadata",
    )
    title = load_catalog_doc_title(doc_path)
    doc_link = relative_catalog_doc_link(catalog_path=catalog_path, doc_path=doc_path)
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
            "status_kind",
        }
        and METADATA_TOKEN_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(f"{metadata_path}: '{field_name}' must use lowercase slug tokens.")
    return value


def _reject_unknown_keys(
    payload: dict[str, object],
    *,
    allowed_keys: frozenset[str],
    label: str,
    metadata_path: Path,
) -> None:
    unknown_keys = sorted(str(key) for key in payload if str(key) not in allowed_keys)
    if unknown_keys:
        raise ValueError(f"{metadata_path}: {label} has unknown key(s): {', '.join(unknown_keys)}.")


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
        _reject_unknown_keys(
            relation_payload,
            allowed_keys=_REGISTRY_RELATION_KEYS,
            label=f"relation {index}",
            metadata_path=metadata_path,
        )
        relation_type = str(relation_payload.get("type") or "").strip()
        target_registry_id = str(relation_payload.get("target") or "").strip()
        if not relation_type:
            raise ValueError(f"{metadata_path}: relation {index} missing 'type'.")
        if relation_type not in ALLOWED_RELATION_TYPES:
            allowed = ", ".join(sorted(ALLOWED_RELATION_TYPES))
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
        if METADATA_TOKEN_PATTERN.fullmatch(tool) is None:
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
        _reject_unknown_keys(
            route_payload,
            allowed_keys=_RELATED_TOOL_ROUTE_KEYS,
            label=f"related_tool_routes[{index}]",
            metadata_path=metadata_path,
        )
        tool = str(route_payload.get("tool") or "").strip()
        route_id = str(route_payload.get("route") or "").strip()
        if not tool:
            raise ValueError(f"{metadata_path}: related_tool_routes[{index}] missing 'tool'.")
        if METADATA_TOKEN_PATTERN.fullmatch(tool) is None:
            raise ValueError(f"{metadata_path}: related_tool_routes[{index}] tool must use lowercase slug tokens.")
        if not route_id:
            raise ValueError(f"{metadata_path}: related_tool_routes[{index}] missing 'route'.")
        if METADATA_TOKEN_PATTERN.fullmatch(route_id) is None:
            raise ValueError(f"{metadata_path}: related_tool_routes[{index}] route must use lowercase slug tokens.")
        route_key = (tool, route_id)
        if route_key in seen:
            raise ValueError(f"{metadata_path}: duplicate related tool route '{tool}/{route_id}'.")
        seen.add(route_key)
        route_refs.append(CatalogProcedureToolRouteReference(tool=tool, route_id=route_id))
    return tuple(route_refs)


def _optional_metadata_keywords(
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
        if TOOL_SOURCE_KEYWORD_PATTERN.fullmatch(keyword) is None:
            raise ValueError(
                f"{metadata_path}: keyword {index} must use lowercase words, spaces, hyphens, or underscores."
            )
        if keyword in seen:
            raise ValueError(f"{metadata_path}: duplicate keyword '{keyword}'.")
        seen.add(keyword)
        keywords.append(keyword)
    return tuple(keywords)


def _optional_tool_source_keywords(
    payload: dict[str, object],
    *,
    metadata_path: Path,
) -> tuple[str, ...]:
    return _optional_metadata_keywords(payload, metadata_path=metadata_path)


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
        _reject_unknown_keys(
            route_payload,
            allowed_keys=_TOOL_SOURCE_ROUTE_KEYS,
            label=f"route {index}",
            metadata_path=metadata_path,
        )
        route_id = str(route_payload.get("id") or "").strip()
        relative_path = str(route_payload.get("path") or "").strip()
        summary = str(route_payload.get("summary") or "").strip()
        if not route_id:
            raise ValueError(f"{metadata_path}: route {index} missing 'id'.")
        if METADATA_TOKEN_PATTERN.fullmatch(route_id) is None:
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
                title=load_catalog_doc_title(resolved_doc_path),
                doc_path=relative_catalog_doc_link(catalog_path=catalog_path, doc_path=resolved_doc_path),
                summary=summary,
            )
        )
    return tuple(routes)
