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

import pytest

from dnadesign.devtools.architecture_boundaries import (
    find_legacy_surface_violations,
    find_undeclared_cross_tool_imports,
    main,
)


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


def test_find_undeclared_cross_tool_imports_reports_clean_absolute_import_target(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "api.py", "from dnadesign.bar.api import run\n")
    _write(tmp_path / "src" / "dnadesign" / "bar" / "api.py", "def run():\n    return 1\n")

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges=set(),
    )

    assert len(violations) == 1
    assert violations[0].import_target == "dnadesign.bar.api"


def test_find_undeclared_cross_tool_imports_allows_ops_to_usr_default_edge(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "ops" / "gates.py", "from dnadesign.usr import Dataset\n")
    _write(tmp_path / "src" / "dnadesign" / "usr" / "__init__.py", "class Dataset:\n    pass\n")

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


def test_find_undeclared_cross_tool_imports_allows_construct_to_usr_default_edge(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "construct" / "runtime.py", "from dnadesign.usr import Dataset\n")
    _write(tmp_path / "src" / "dnadesign" / "usr" / "__init__.py", "class Dataset:\n    pass\n")

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


def test_find_undeclared_cross_tool_imports_allows_ops_to_infer_default_edge(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "plan.py",
        "from dnadesign.infer import validate_runbook_gpu_resources\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "infer" / "__init__.py",
        "def validate_runbook_gpu_resources(**_kwargs):\n    return None\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


@pytest.mark.parametrize(
    ("owner_tool", "imported_tool"),
    (
        ("notify", "construct"),
        ("notify", "densegen"),
        ("notify", "infer"),
        ("ops", "construct"),
        ("ops", "densegen"),
        ("ops", "notify"),
    ),
)
def test_find_undeclared_cross_tool_imports_allows_contract_owner_edges(
    tmp_path: Path,
    owner_tool: str,
    imported_tool: str,
) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / owner_tool / "api.py",
        f"from dnadesign.{imported_tool}.contracts import run\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / imported_tool / "contracts.py",
        "def run():\n    return 1\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


def test_find_undeclared_cross_tool_imports_rejects_internal_src_target_even_for_allowed_edge(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "foo" / "api.py",
        "from dnadesign.bar.src.runtime import run\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "bar" / "src" / "runtime.py", "def run():\n    return 1\n")

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges={("foo", "bar")},
    )

    assert len(violations) == 1
    assert violations[0].owner_tool == "foo"
    assert violations[0].imported_tool == "bar"
    assert violations[0].import_target == "dnadesign.bar.src.runtime"


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


def test_find_legacy_surface_violations_flags_removed_repo_root_contract_paths(tmp_path: Path) -> None:
    (tmp_path / "src" / "dnadesign" / "_contracts").mkdir(parents=True, exist_ok=True)
    legacy_usr_roots = tmp_path / "src" / "dnadesign" / "usr_roots.py"
    legacy_usr_roots.parent.mkdir(parents=True, exist_ok=True)
    legacy_usr_roots.write_text("# legacy\n", encoding="utf-8")

    violations = find_legacy_surface_violations(repo_root=tmp_path)

    assert [item.path.relative_to(tmp_path).as_posix() for item in violations] == [
        "src/dnadesign/_contracts",
        "src/dnadesign/usr_roots.py",
    ]


def test_find_legacy_surface_violations_flags_removed_ops_study_paths(tmp_path: Path) -> None:
    legacy_path = tmp_path / "src" / "dnadesign" / "ops" / "promoter_preflight_coordinator.py"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text("# legacy study-owned surface\n", encoding="utf-8")
    unexpected_cli_path = tmp_path / "src" / "dnadesign" / "ops" / "legacy_cli_bridge.py"
    unexpected_cli_path.write_text("# removed cli bridge\n", encoding="utf-8")

    violations = find_legacy_surface_violations(repo_root=tmp_path)

    assert [item.path.relative_to(tmp_path).as_posix() for item in violations] == [
        "src/dnadesign/ops/legacy_cli_bridge.py",
        "src/dnadesign/ops/promoter_preflight_coordinator.py",
    ]


def test_find_legacy_surface_violations_accepts_relative_repo_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_cli_path = tmp_path / "src" / "dnadesign" / "ops" / "legacy_cli_bridge.py"
    legacy_cli_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_cli_path.write_text("# removed cli bridge\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    violations = find_legacy_surface_violations(repo_root=Path("."))

    assert [item.path.relative_to(tmp_path).as_posix() for item in violations] == [
        "src/dnadesign/ops/legacy_cli_bridge.py",
    ]


def test_main_fails_when_legacy_contract_surface_paths_exist(tmp_path: Path) -> None:
    (tmp_path / "src" / "dnadesign" / "foo").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "dnadesign" / "bar").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "dnadesign" / "_contracts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "dnadesign" / "foo" / "api.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    (tmp_path / "src" / "dnadesign" / "bar" / "api.py").write_text("def run():\n    return 1\n", encoding="utf-8")

    rc = main(["--repo-root", str(tmp_path)])

    assert rc == 1


def test_main_fails_on_syntax_error_in_checked_file(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "api.py", "def broken(:\n")
    _write(tmp_path / "src" / "dnadesign" / "bar" / "api.py", "def run():\n    return 1\n")

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1
