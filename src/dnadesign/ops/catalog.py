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
import re
from dataclasses import dataclass
from pathlib import Path

_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


@dataclass(frozen=True)
class CatalogProcedureEntry:
    registry_id: str
    title: str
    doc_path: str
    entry_type: str
    plane: str
    execution_kind: str
    progress_kind: str
    summary: str


@dataclass(frozen=True)
class CatalogToolSourceEntry:
    tool: str
    title: str
    doc_path: str
    summary: str


@dataclass(frozen=True)
class CatalogQuery:
    query: str | None = None
    entry_type: str | None = None
    plane: str | None = None
    execution_kind: str | None = None
    progress_kind: str | None = None
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
class RunbookCatalog:
    repo_root: Path
    catalog_path: Path
    procedures: tuple[CatalogProcedureEntry, ...]
    tool_sources: tuple[CatalogToolSourceEntry, ...]

    def find_procedure(self, registry_id: str) -> CatalogProcedureEntry | None:
        for entry in self.procedures:
            if entry.registry_id == registry_id:
                return entry
        return None


def resolve_catalog_doc_path(*, catalog_path: Path, doc_path: str) -> Path:
    return (catalog_path.parent / doc_path).resolve()


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

    text = catalog_path.read_text(encoding="utf-8")
    procedure_rows = _extract_table_rows(text=text, heading="### Authoritative cross-tool procedures")
    tool_rows = _extract_table_rows(text=text, heading="### Tool-local runbook sources")

    procedures = tuple(_parse_procedure_row(row) for row in procedure_rows)
    tool_sources = tuple(_parse_tool_source_row(row) for row in tool_rows)
    return RunbookCatalog(
        repo_root=resolved_repo_root,
        catalog_path=catalog_path,
        procedures=procedures,
        tool_sources=tool_sources,
    )


def filter_runbook_catalog(
    catalog: RunbookCatalog,
    *,
    query: CatalogQuery,
) -> tuple[tuple[CatalogProcedureEntry, ...], tuple[CatalogToolSourceEntry, ...]]:
    query_tokens = query.query_tokens()
    procedures = tuple(
        entry
        for entry in catalog.procedures
        if _matches_procedure_query(entry=entry, query=query, query_tokens=query_tokens)
    )
    tool_sources = tuple(
        entry
        for entry in catalog.tool_sources
        if _matches_tool_source_query(entry=entry, query=query, query_tokens=query_tokens)
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


def _parse_procedure_row(row: list[str]) -> CatalogProcedureEntry:
    if len(row) != 7:
        raise ValueError(f"expected 7 columns for procedure row, got {len(row)}: {row}")
    title, doc_path = _parse_link_cell(row[1])
    return CatalogProcedureEntry(
        registry_id=_strip_ticks(row[0]),
        title=title,
        doc_path=doc_path,
        entry_type=_strip_ticks(row[2]),
        plane=_strip_ticks(row[3]),
        execution_kind=_strip_ticks(row[4]),
        progress_kind=_strip_ticks(row[5]),
        summary=row[6].strip(),
    )


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


def _matches_procedure_query(
    *,
    entry: CatalogProcedureEntry,
    query: CatalogQuery,
    query_tokens: tuple[str, ...],
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
    if query_tokens and not _query_tokens_match(_procedure_haystack(entry), query_tokens):
        return False
    return True


def _matches_tool_source_query(
    *,
    entry: CatalogToolSourceEntry,
    query: CatalogQuery,
    query_tokens: tuple[str, ...],
) -> bool:
    if query.has_procedure_filters():
        return False
    if query.tool is not None and entry.tool != query.tool:
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
            entry.execution_kind,
            entry.progress_kind,
            entry.summary,
        )
    ).lower()


def _tool_source_haystack(entry: CatalogToolSourceEntry) -> str:
    return " ".join((entry.tool, entry.title, entry.doc_path, entry.summary)).lower()
