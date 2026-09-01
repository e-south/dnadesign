"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/docs_contract/test_notebook.py

USR notebook setup and display contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
import tomllib

from .helpers import read_text, repo_root


def test_notebook_agent_setup_uses_a_declared_locked_environment() -> None:
    agent_text = read_text("src/dnadesign/usr/notebooks/AGENTS.md")
    project = tomllib.loads(read_text("pyproject.toml"))

    assert "uv sync --locked" in agent_text
    assert "--group notebooks" not in agent_text
    assert any(requirement.startswith("marimo>=") for requirement in project["project"]["dependencies"])


def test_explorer_plot_cell_replaces_the_displayed_output() -> None:
    notebook_path = repo_root() / "src/dnadesign/usr/notebooks/usr_explorer.py"
    tree = ast.parse(notebook_path.read_text(encoding="utf-8"))

    output_replace_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "replace"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "output"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "mo"
    ]

    assert len(output_replace_calls) == 1
    displayed = output_replace_calls[0].args[0]
    assert isinstance(displayed, ast.IfExp)
    assert _qualified_call_name(displayed.body) == "mo.ui.tabs"
    assert _qualified_call_name(displayed.orelse) == "mo.md"


def test_explorer_does_not_export_private_names_between_cells() -> None:
    notebook_path = repo_root() / "src/dnadesign/usr/notebooks/usr_explorer.py"
    tree = ast.parse(notebook_path.read_text(encoding="utf-8"))

    for cell in (node for node in tree.body if isinstance(node, ast.FunctionDef)):
        returned_names = {
            item.id
            for node in ast.walk(cell)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple)
            for item in node.value.elts
            if isinstance(item, ast.Name)
        }
        assert not {name for name in returned_names if name.startswith("_")}


def _qualified_call_name(node: ast.expr) -> str | None:
    if not isinstance(node, ast.Call):
        return None

    parts: list[str] = []
    current: ast.expr = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))
