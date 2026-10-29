"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/docs/test_check_architecture.py

Architecture contract for the documentation checker.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
from pathlib import Path

DOCS_PACKAGE = Path(__file__).parents[2] / "docs"


def _defined_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


def test_docs_checker_is_an_ordered_coordinator_not_a_policy_monolith() -> None:
    coordinator = DOCS_PACKAGE / "checks.py"
    responsibilities = {
        "markdown_inventory.py": "_find_broken_links",
        "document_metadata.py": "_find_tool_docs_metadata_issues",
        "operations_contracts.py": "_find_operational_runbook_path_issues",
        "public_surface_contracts.py": "_find_tool_readme_structure_issues",
        "banner_contracts.py": "_find_tool_readme_banner_issues",
    }

    assert len(coordinator.read_text(encoding="utf-8").splitlines()) <= 550
    for module_name, representative_function in responsibilities.items():
        module_path = DOCS_PACKAGE / module_name
        assert module_path.is_file(), f"missing docs-check responsibility module: {module_name}"
        assert representative_function in _defined_functions(module_path)
