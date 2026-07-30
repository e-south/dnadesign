"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/catalog/paths.py

Path resolution helpers for the Ops runbook catalog.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path

from .constants import REGISTRY_METADATA_SUFFIX, TITLE_HEADING_PATTERN


def resolve_catalog_doc_path(*, catalog_path: Path, doc_path: str) -> Path:
    return (catalog_path.parent / doc_path).resolve()


def resolve_registry_metadata_path_for_doc_path(doc_path: Path | str) -> Path:
    normalized = Path(doc_path)
    if normalized.parent.name == "contracts":
        return normalized.parent / "registry" / f"{normalized.stem}{REGISTRY_METADATA_SUFFIX}"
    return normalized.with_name(f"{normalized.stem}{REGISTRY_METADATA_SUFFIX}")


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


def catalog_metadata_search_roots(repo_root: Path) -> tuple[Path, ...]:
    """Return the checked-in documentation roots that may publish catalog metadata."""

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


def resolve_doc_path_for_metadata(*, metadata_path: Path, repo_root: Path) -> Path:
    relative_metadata = metadata_path.resolve().relative_to(repo_root.resolve())
    if relative_metadata.parent.name == "registry":
        doc_relative = relative_metadata.parent.parent / (
            relative_metadata.name[: -len(REGISTRY_METADATA_SUFFIX)] + ".md"
        )
        resolved_doc_path = (repo_root / doc_relative).resolve()
        if resolved_doc_path.exists():
            return resolved_doc_path
    return resolve_doc_path_for_sidecar(
        metadata_path=metadata_path,
        repo_root=repo_root,
        suffix=REGISTRY_METADATA_SUFFIX,
        error_prefix="registry metadata",
    )


def resolve_doc_path_for_sidecar(
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


def relative_catalog_doc_link(*, catalog_path: Path, doc_path: Path) -> str:
    return os.path.relpath(doc_path, start=catalog_path.parent).replace(os.sep, "/")


def load_catalog_doc_title(doc_path: Path) -> str:
    text = doc_path.read_text(encoding="utf-8")
    match = TITLE_HEADING_PATTERN.search(text)
    if match is None:
        raise ValueError(f"{doc_path}: missing top-level markdown heading.")
    title = match.group(1).strip()
    if not title:
        raise ValueError(f"{doc_path}: top-level markdown heading must not be empty.")
    return title
