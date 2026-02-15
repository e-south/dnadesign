"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/config/test_internal_catalog_usage.py

Guards internal app/cli code against using removed motif_store config aliases.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
from pathlib import Path


def _motif_store_alias_accesses() -> list[str]:
    cruncher_root = Path(__file__).resolve().parents[2]
    roots = [cruncher_root / "src" / "app", cruncher_root / "src" / "cli"]
    hits: list[str] = []
    for root in roots:
        for file_path in sorted(root.rglob("*.py")):
            text = file_path.read_text()
            tree = ast.parse(text, filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr == "motif_store":
                    hits.append(f"{file_path.relative_to(cruncher_root)}:{node.lineno}")
    return hits


def test_app_and_cli_use_catalog_config_not_motif_store_alias() -> None:
    hits = _motif_store_alias_accesses()
    assert not hits, "Replace internal motif_store alias access with catalog access:\n" + "\n".join(hits)


def _motif_discovery_alias_accesses() -> list[str]:
    cruncher_root = Path(__file__).resolve().parents[2]
    roots = [cruncher_root / "src" / "app", cruncher_root / "src" / "cli"]
    hits: list[str] = []
    for root in roots:
        for file_path in sorted(root.rglob("*.py")):
            text = file_path.read_text()
            tree = ast.parse(text, filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr == "motif_discovery":
                    hits.append(f"{file_path.relative_to(cruncher_root)}:{node.lineno}")
    return hits


def test_app_and_cli_use_discover_config_not_motif_discovery_alias() -> None:
    hits = _motif_discovery_alias_accesses()
    assert not hits, "Replace internal motif_discovery alias access with discover access:\n" + "\n".join(hits)
