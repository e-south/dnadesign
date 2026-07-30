"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_discovery_performance.py

Parity and traversal-boundary tests for OPS metadata discovery.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

import dnadesign.ops.catalog.loader as catalog_loader
from dnadesign.ops.catalog.constants import REGISTRY_METADATA_SUFFIX, TOOL_SOURCE_METADATA_SUFFIX
from dnadesign.ops.catalog.metadata import discover_catalog_metadata_paths
from dnadesign.ops.catalog.paths import catalog_metadata_search_roots
from dnadesign.ops.discovery import discover_named_files
from dnadesign.ops.status.registry_loader import _iter_status_registry_fragment_paths_for_root


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def test_catalog_discovery_preserves_existing_file_parity() -> None:
    repo_root = _repo_root()
    search_roots = catalog_metadata_search_roots(repo_root)
    expected_registry = tuple(
        sorted(
            path.resolve()
            for search_root in search_roots
            for path in search_root.rglob(f"*{REGISTRY_METADATA_SUFFIX}")
            if not any(segment in {"archived", "prototypes", "__pycache__"} for segment in path.parts)
        )
    )
    expected_tool_sources = tuple(
        sorted(
            path.resolve()
            for search_root in search_roots
            for path in search_root.rglob(f"*{TOOL_SOURCE_METADATA_SUFFIX}")
            if not any(segment in {"archived", "prototypes", "__pycache__"} for segment in path.parts)
        )
    )

    discovered = discover_catalog_metadata_paths(repo_root)

    assert discovered.registry_paths == expected_registry
    assert discovered.tool_source_paths == expected_tool_sources


def test_catalog_loader_discovers_both_sidecar_kinds_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    original = catalog_loader.discover_catalog_metadata_paths

    def _discover(repo_root: Path):
        nonlocal calls
        calls += 1
        return original(repo_root)

    monkeypatch.setattr(catalog_loader, "discover_catalog_metadata_paths", _discover)

    catalog = catalog_loader.load_runbook_catalog(repo_root=_repo_root())

    assert catalog.procedures
    assert catalog.tool_sources
    assert calls == 1


def test_catalog_discovery_keeps_malformed_sidecars_visible_to_validation(tmp_path: Path) -> None:
    catalog_path = tmp_path / "docs" / "runbooks" / "README.md"
    doc_path = tmp_path / "docs" / "operations" / "demo.md"
    metadata_path = doc_path.with_name("demo.registry.yaml")
    catalog_path.parent.mkdir(parents=True)
    doc_path.parent.mkdir(parents=True)
    catalog_path.write_text("# Runbooks\n", encoding="utf-8")
    doc_path.write_text("# Demo\n", encoding="utf-8")
    metadata_path.write_text("schema_version: 1\nlegacy_alias: hidden-error\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"registry metadata has unknown key\(s\): legacy_alias"):
        catalog_loader.load_runbook_catalog(repo_root=tmp_path)


def test_status_discovery_preserves_existing_file_parity() -> None:
    dnadesign_root = _repo_root() / "src" / "dnadesign"
    expected = tuple(
        sorted(
            path
            for path in dnadesign_root.rglob("status.registry.yaml")
            if path.is_file()
            and (
                path.parent.name == "ops"
                or (
                    len(path.resolve().relative_to(dnadesign_root).parts) == 4
                    and path.resolve().relative_to(dnadesign_root).parts[0:2] == ("ops", "providers")
                )
            )
        )
    )

    assert _iter_status_registry_fragment_paths_for_root(dnadesign_root=dnadesign_root) == expected


def test_named_file_discovery_prunes_generated_and_archived_trees(tmp_path: Path) -> None:
    visible = tmp_path / "tool" / "ops" / "status.registry.yaml"
    generated = tmp_path / "tool" / "outputs" / "ops" / "status.registry.yaml"
    archived = tmp_path / "tool" / "archived" / "ops" / "status.registry.yaml"
    for path in (visible, generated, archived):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("version: 1\n", encoding="utf-8")

    discovered = discover_named_files(
        roots=(tmp_path,),
        names=frozenset({"status.registry.yaml"}),
    )

    assert discovered == (visible.resolve(),)
