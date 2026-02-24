"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/test_architecture_boundaries.py

Tests for cross-tool import boundary checks used in CI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.devtools.architecture_boundaries import find_undeclared_cross_tool_imports, main


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_find_undeclared_cross_tool_imports_allows_declared_edge(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "api.py", "from dnadesign.bar.api import run\n")
    _write(tmp_path / "src" / "dnadesign" / "bar" / "api.py", "def run():\n    return 1\n")

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges={("foo", "bar")},
    )

    assert violations == []


def test_find_undeclared_cross_tool_imports_reports_undeclared_edge(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "api.py", "from dnadesign.bar.api import run\n")
    _write(tmp_path / "src" / "dnadesign" / "bar" / "api.py", "def run():\n    return 1\n")

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges=set(),
    )

    assert len(violations) == 1
    assert violations[0].owner_tool == "foo"
    assert violations[0].imported_tool == "bar"


def test_find_undeclared_cross_tool_imports_reports_relative_cross_tool_edge(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "subpkg" / "api.py", "from ...bar.api import run\n")
    _write(tmp_path / "src" / "dnadesign" / "bar" / "api.py", "def run():\n    return 1\n")

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges=set(),
    )

    assert len(violations) == 1
    assert violations[0].owner_tool == "foo"
    assert violations[0].imported_tool == "bar"


def test_find_undeclared_cross_tool_imports_allows_relative_within_tool(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "subpkg" / "api.py", "from ..core.api import run\n")
    _write(tmp_path / "src" / "dnadesign" / "foo" / "core" / "api.py", "def run():\n    return 1\n")
    _write(tmp_path / "src" / "dnadesign" / "bar" / "api.py", "def run():\n    return 1\n")

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges=set(),
    )

    assert violations == []


def test_find_undeclared_cross_tool_imports_reports_relative_import_without_module(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "subpkg" / "api.py", "from ... import bar\n")
    _write(tmp_path / "src" / "dnadesign" / "bar" / "__init__.py", "")

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges=set(),
    )

    assert len(violations) == 1
    assert violations[0].owner_tool == "foo"
    assert violations[0].imported_tool == "bar"


def test_find_undeclared_cross_tool_imports_allows_relative_import_without_module_within_tool(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "subpkg" / "api.py", "from .. import core\n")
    _write(tmp_path / "src" / "dnadesign" / "foo" / "core.py", "def run():\n    return 1\n")
    _write(tmp_path / "src" / "dnadesign" / "bar" / "__init__.py", "")

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges=set(),
    )

    assert violations == []


def test_find_undeclared_cross_tool_imports_ignores_test_files(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "tests" / "test_api.py", "from dnadesign.bar import api\n")
    _write(tmp_path / "src" / "dnadesign" / "bar" / "api.py", "def run():\n    return 1\n")

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges=set(),
    )

    assert violations == []


def test_find_undeclared_cross_tool_imports_ignores_archived_and_prototypes(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "api.py", "def run():\n    return 1\n")
    _write(tmp_path / "src" / "dnadesign" / "bar" / "api.py", "def run():\n    return 1\n")
    _write(
        tmp_path / "src" / "dnadesign" / "archived" / "legacy.py",
        "from dnadesign.bar.api import run\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "prototypes" / "draft.py",
        "from dnadesign.foo.api import run\n",
    )

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges=set(),
    )

    assert violations == []


def test_main_fails_on_syntax_error_in_checked_file(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "api.py", "def broken(:\n")
    _write(tmp_path / "src" / "dnadesign" / "bar" / "api.py", "def run():\n    return 1\n")

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1
