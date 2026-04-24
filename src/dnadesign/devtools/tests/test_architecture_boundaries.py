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
    TOP_LEVEL_LEGACY_DIRECTORIES,
    TOP_LEVEL_ROOT_MODULES,
    TOP_LEVEL_SHARED_INFRA_PACKAGES,
    TOP_LEVEL_TOOL_BOUNDARY_PACKAGES,
    find_legacy_surface_violations,
    find_top_level_layout_violations,
    find_undeclared_cross_tool_imports,
    main,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _scaffold_top_level_layout(tmp_path: Path) -> None:
    src_root = tmp_path / "src" / "dnadesign"
    src_root.mkdir(parents=True, exist_ok=True)
    for name in TOP_LEVEL_ROOT_MODULES:
        _write(src_root / name, "")
    for name in TOP_LEVEL_TOOL_BOUNDARY_PACKAGES | TOP_LEVEL_SHARED_INFRA_PACKAGES:
        _write(src_root / name / "__init__.py", "")
    for name in TOP_LEVEL_LEGACY_DIRECTORIES:
        (src_root / name).mkdir(parents=True, exist_ok=True)


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


def test_find_undeclared_cross_tool_imports_rejects_ops_to_studies_edge(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "ops" / "gates.py", "from dnadesign.studies import StudyOpsContract\n")
    _write(tmp_path / "src" / "dnadesign" / "studies" / "__init__.py", "class StudyOpsContract:\n    pass\n")

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert len(violations) == 1
    assert violations[0].owner_tool == "ops"
    assert violations[0].imported_tool == "studies"


def test_find_undeclared_cross_tool_imports_allows_construct_to_usr_default_edge(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "construct" / "runtime.py", "from dnadesign.usr import Dataset\n")
    _write(tmp_path / "src" / "dnadesign" / "usr" / "__init__.py", "class Dataset:\n    pass\n")

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


def test_find_undeclared_cross_tool_imports_allows_latentdna_to_usr_default_edge(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "latentdna" / "runtime.py", "from dnadesign.usr import Dataset\n")
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


def test_find_undeclared_cross_tool_imports_allows_studies_to_densegen_public_edge(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "studies" / "families" / "demo" / "surface.py",
        "from dnadesign.densegen import inspect_analysis_surface\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "densegen" / "__init__.py",
        "def inspect_analysis_surface(_path):\n    return None\n",
    )

    violations = find_undeclared_cross_tool_imports(repo_root=tmp_path)

    assert violations == []


@pytest.mark.parametrize(
    ("owner_tool", "imported_tool"),
    (
        ("baserender", "contracts"),
        ("cruncher", "contracts"),
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


def test_find_undeclared_cross_tool_imports_rejects_testsupport_imports_from_runtime_code(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "api.py", "from dnadesign.testsupport.usr import ensure_registry\n")
    _write(tmp_path / "src" / "dnadesign" / "testsupport" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "testsupport" / "usr.py", "def ensure_registry():\n    return None\n")

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges=set(),
    )

    assert len(violations) == 1
    assert violations[0].owner_tool == "foo"
    assert violations[0].imported_tool == "testsupport"
    assert violations[0].import_target == "dnadesign.testsupport.usr"


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


def test_find_undeclared_cross_tool_imports_scans_study_family_modules(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "foo" / "api.py", "def run():\n    return 1\n")
    _write(tmp_path / "src" / "dnadesign" / "bar" / "src" / "runtime.py", "def run():\n    return 1\n")
    _write(
        tmp_path / "src" / "dnadesign" / "studies" / "families" / "demo" / "status.py",
        "from dnadesign.bar.src.runtime import run\n",
    )

    violations = find_undeclared_cross_tool_imports(
        repo_root=tmp_path,
        allowed_edges=set(),
    )

    assert len(violations) == 1
    assert violations[0].owner_tool == "studies"
    assert violations[0].imported_tool == "bar"
    assert violations[0].import_target == "dnadesign.bar.src.runtime"


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


def test_find_legacy_surface_violations_ignores_cache_only_legacy_directory(tmp_path: Path) -> None:
    cache_dir = tmp_path / "src" / "dnadesign" / "_contracts" / "__pycache__"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "__init__.cpython-312.pyc").write_bytes(b"cache")

    violations = find_legacy_surface_violations(repo_root=tmp_path)

    assert violations == []


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


def test_find_top_level_layout_violations_accepts_expected_inventory(tmp_path: Path) -> None:
    _scaffold_top_level_layout(tmp_path)

    violations = find_top_level_layout_violations(repo_root=tmp_path)

    assert violations == []


def test_find_top_level_layout_violations_flags_unexpected_top_level_directory(tmp_path: Path) -> None:
    _scaffold_top_level_layout(tmp_path)
    unexpected_dir = tmp_path / "src" / "dnadesign" / "scratchpad"
    unexpected_dir.mkdir(parents=True, exist_ok=True)

    violations = find_top_level_layout_violations(repo_root=tmp_path)

    assert [(item.reason, item.path.relative_to(tmp_path).as_posix()) for item in violations] == [
        ("unexpected top-level directory", "src/dnadesign/scratchpad"),
    ]


def test_find_top_level_layout_violations_flags_unexpected_top_level_module(tmp_path: Path) -> None:
    _scaffold_top_level_layout(tmp_path)
    _write(tmp_path / "src" / "dnadesign" / "helpers.py", "# drift\n")

    violations = find_top_level_layout_violations(repo_root=tmp_path)

    assert [(item.reason, item.path.relative_to(tmp_path).as_posix()) for item in violations] == [
        ("unexpected top-level module", "src/dnadesign/helpers.py"),
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
