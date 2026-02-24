"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/test_ci_changes.py

Tests for CI change-scope detection used for core/external integration lane routing.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.devtools.ci_changes import (
    determine_scope,
    discover_external_integration_tools,
    discover_repo_tools,
    main,
)


def test_determine_scope_non_pr_runs_full_core_and_external_integration() -> None:
    result = determine_scope(
        event_name="push",
        changed_files=[],
        tool_names={"densegen", "usr"},
        external_integration_tool_names={"densegen"},
    )

    assert result.run_full_core is True
    assert result.run_external_integration is True
    assert result.affected_tools == ["densegen", "usr"]
    assert result.external_integration_tools == ["densegen"]


def test_determine_scope_pr_scopes_single_tool() -> None:
    result = determine_scope(
        event_name="pull_request",
        changed_files=["src/dnadesign/usr/src/cli.py"],
        tool_names={"densegen", "usr"},
        external_integration_tool_names={"densegen"},
    )

    assert result.run_full_core is False
    assert result.run_external_integration is False
    assert result.affected_tools == ["usr"]
    assert result.external_integration_tools == []


def test_determine_scope_pr_densegen_change_triggers_external_integration_lane() -> None:
    result = determine_scope(
        event_name="pull_request",
        changed_files=["src/dnadesign/densegen/src/cli/main.py"],
        tool_names={"densegen", "usr"},
        external_integration_tool_names={"densegen"},
    )

    assert result.run_full_core is False
    assert result.run_external_integration is True
    assert result.affected_tools == ["densegen"]
    assert result.external_integration_tools == ["densegen"]


def test_determine_scope_pr_lockfile_change_triggers_full_core_and_external_integration() -> None:
    result = determine_scope(
        event_name="pull_request",
        changed_files=["pyproject.toml"],
        tool_names={"densegen", "usr"},
        external_integration_tool_names={"densegen"},
    )

    assert result.run_full_core is True
    assert result.run_external_integration is True
    assert result.affected_tools == ["densegen", "usr"]
    assert result.external_integration_tools == ["densegen"]


def test_determine_scope_pr_docs_only_skips_tool_coverage_gate() -> None:
    result = determine_scope(
        event_name="pull_request",
        changed_files=["docs/README.md"],
        tool_names={"densegen", "usr"},
        external_integration_tool_names={"densegen"},
    )

    assert result.run_full_core is False
    assert result.run_external_integration is False
    assert result.affected_tools == []
    assert result.external_integration_tools == []
    assert result.run_coverage_gate is False


def test_determine_scope_pr_shared_code_change_triggers_full_core_and_external_integration() -> None:
    result = determine_scope(
        event_name="pull_request",
        changed_files=["src/dnadesign/devtools/tool_coverage.py"],
        tool_names={"densegen", "usr"},
        external_integration_tool_names={"densegen"},
    )

    assert result.run_full_core is True
    assert result.run_external_integration is True
    assert result.affected_tools == ["densegen", "usr"]
    assert result.external_integration_tools == ["densegen"]


def test_determine_scope_pr_shared_package_root_change_triggers_full_core_and_external_integration() -> None:
    result = determine_scope(
        event_name="pull_request",
        changed_files=["src/dnadesign/__init__.py"],
        tool_names={"densegen", "usr"},
        external_integration_tool_names={"densegen"},
    )

    assert result.run_full_core is True
    assert result.run_external_integration is True
    assert result.affected_tools == ["densegen", "usr"]
    assert result.external_integration_tools == ["densegen"]


def test_determine_scope_pr_external_integration_tool_is_not_densegen() -> None:
    result = determine_scope(
        event_name="pull_request",
        changed_files=["src/dnadesign/cruncher/src/cli.py"],
        tool_names={"cruncher", "densegen", "usr"},
        external_integration_tool_names={"cruncher", "densegen"},
    )

    assert result.run_full_core is False
    assert result.run_external_integration is True
    assert result.affected_tools == ["cruncher"]
    assert result.external_integration_tools == ["cruncher"]


def test_determine_scope_pr_baseline_file_triggers_full_core() -> None:
    result = determine_scope(
        event_name="pull_request",
        changed_files=[".github/tool-coverage-baseline.json"],
        tool_names={"densegen", "usr"},
        external_integration_tool_names={"densegen"},
    )

    assert result.run_full_core is True
    assert result.run_external_integration is False
    assert result.affected_tools == ["densegen", "usr"]
    assert result.external_integration_tools == ["densegen"]


def test_determine_scope_disables_heavy_when_no_external_integration_tools_exist() -> None:
    result = determine_scope(
        event_name="push",
        changed_files=[],
        tool_names={"usr"},
        external_integration_tool_names=set(),
    )

    assert result.run_full_core is True
    assert result.run_external_integration is False
    assert result.affected_tools == ["usr"]
    assert result.external_integration_tools == []


def test_discover_external_integration_tools_detects_marked_test_files(tmp_path: Path) -> None:
    src_root = tmp_path / "src" / "dnadesign"
    foo_test = src_root / "foo" / "tests" / "test_one.py"
    bar_test = src_root / "bar" / "tests" / "test_two.py"
    foo_test.parent.mkdir(parents=True, exist_ok=True)
    bar_test.parent.mkdir(parents=True, exist_ok=True)
    foo_test.write_text("import pytest\npytestmark = pytest.mark.integration\n", encoding="utf-8")
    bar_test.write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    external_integration_tools = discover_external_integration_tools(
        repo_root=tmp_path,
        tool_names={"foo", "bar"},
    )

    assert external_integration_tools == {"foo"}


def test_discover_external_integration_tools_detects_marked_conftest(tmp_path: Path) -> None:
    src_root = tmp_path / "src" / "dnadesign"
    conftest = src_root / "foo" / "tests" / "conftest.py"
    test_file = src_root / "foo" / "tests" / "test_one.py"
    conftest.parent.mkdir(parents=True, exist_ok=True)
    conftest.write_text("import pytest\npytestmark = pytest.mark.fimo\n", encoding="utf-8")
    test_file.write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    external_integration_tools = discover_external_integration_tools(
        repo_root=tmp_path,
        tool_names={"foo"},
    )

    assert external_integration_tools == {"foo"}


def test_discover_repo_tools_excludes_devtools_and_dunder_dirs(tmp_path: Path) -> None:
    src_root = tmp_path / "src" / "dnadesign"
    (src_root / "usr").mkdir(parents=True, exist_ok=True)
    (src_root / "densegen").mkdir(parents=True, exist_ok=True)
    (src_root / "devtools").mkdir(parents=True, exist_ok=True)
    (src_root / "__pycache__").mkdir(parents=True, exist_ok=True)
    (src_root / "archived").mkdir(parents=True, exist_ok=True)
    (src_root / "prototypes").mkdir(parents=True, exist_ok=True)

    tool_names = discover_repo_tools(repo_root=tmp_path)
    assert tool_names == {"densegen", "usr"}


def test_discover_external_integration_tools_ignores_marker_text_in_comments(tmp_path: Path) -> None:
    src_root = tmp_path / "src" / "dnadesign"
    foo_test = src_root / "foo" / "tests" / "test_one.py"
    foo_test.parent.mkdir(parents=True, exist_ok=True)
    foo_test.write_text(
        '"""Example string containing pytest.mark.integration."""\n'
        "# pytest.mark.fimo\n"
        "def test_ok():\n"
        "    assert True\n",
        encoding="utf-8",
    )

    external_integration_tools = discover_external_integration_tools(
        repo_root=tmp_path,
        tool_names={"foo"},
    )

    assert external_integration_tools == set()


def test_discover_external_integration_tools_detects_function_decorator_markers(tmp_path: Path) -> None:
    src_root = tmp_path / "src" / "dnadesign"
    foo_test = src_root / "foo" / "tests" / "test_one.py"
    foo_test.parent.mkdir(parents=True, exist_ok=True)
    foo_test.write_text(
        "import pytest\n\n@pytest.mark.fimo\ndef test_ok():\n    assert True\n",
        encoding="utf-8",
    )

    external_integration_tools = discover_external_integration_tools(
        repo_root=tmp_path,
        tool_names={"foo"},
    )

    assert external_integration_tools == {"foo"}


def test_discover_external_integration_tools_detects_markers_with_pytest_alias(tmp_path: Path) -> None:
    src_root = tmp_path / "src" / "dnadesign"
    foo_test = src_root / "foo" / "tests" / "test_one.py"
    foo_test.parent.mkdir(parents=True, exist_ok=True)
    foo_test.write_text("import pytest as pt\n\npytestmark = pt.mark.integration\n", encoding="utf-8")

    external_integration_tools = discover_external_integration_tools(
        repo_root=tmp_path,
        tool_names={"foo"},
    )

    assert external_integration_tools == {"foo"}


def test_discover_external_integration_tools_detects_markers_with_mark_import_alias(tmp_path: Path) -> None:
    src_root = tmp_path / "src" / "dnadesign"
    foo_test = src_root / "foo" / "tests" / "test_one.py"
    foo_test.parent.mkdir(parents=True, exist_ok=True)
    foo_test.write_text(
        "from pytest import mark as m\n\n@m.fimo\ndef test_ok():\n    assert True\n",
        encoding="utf-8",
    )

    external_integration_tools = discover_external_integration_tools(
        repo_root=tmp_path,
        tool_names={"foo"},
    )

    assert external_integration_tools == {"foo"}


def test_discover_external_integration_tools_detects_param_level_markers(tmp_path: Path) -> None:
    src_root = tmp_path / "src" / "dnadesign"
    foo_test = src_root / "foo" / "tests" / "test_one.py"
    foo_test.parent.mkdir(parents=True, exist_ok=True)
    foo_test.write_text(
        "import pytest\n\n"
        "@pytest.mark.parametrize('x', [pytest.param(1, marks=pytest.mark.integration)])\n"
        "def test_ok(x):\n"
        "    assert x == 1\n",
        encoding="utf-8",
    )

    external_integration_tools = discover_external_integration_tools(
        repo_root=tmp_path,
        tool_names={"foo"},
    )

    assert external_integration_tools == {"foo"}


def test_main_fails_when_changed_files_input_is_missing(tmp_path: Path) -> None:
    src_root = tmp_path / "src" / "dnadesign" / "usr" / "tests"
    src_root.mkdir(parents=True, exist_ok=True)
    (src_root / "test_a.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")

    baseline_path = tmp_path / "tool-coverage-baseline.json"
    baseline_path.write_text('{"usr": 0.0}', encoding="utf-8")

    rc = main(
        [
            "--event-name",
            "pull_request",
            "--repo-root",
            str(tmp_path),
            "--baseline-json",
            str(baseline_path),
            "--changed-files-file",
            str(tmp_path / "missing.txt"),
        ]
    )

    assert rc == 1


def test_main_fails_when_pr_changed_files_are_empty(tmp_path: Path) -> None:
    usr_tests = tmp_path / "src" / "dnadesign" / "usr" / "tests"
    usr_tests.mkdir(parents=True, exist_ok=True)
    (usr_tests / "test_a.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")

    baseline_path = tmp_path / "tool-coverage-baseline.json"
    baseline_path.write_text('{"usr": 0.0}', encoding="utf-8")

    changed_files = tmp_path / "changed.txt"
    changed_files.write_text("", encoding="utf-8")

    rc = main(
        [
            "--event-name",
            "pull_request",
            "--repo-root",
            str(tmp_path),
            "--baseline-json",
            str(baseline_path),
            "--changed-files-file",
            str(changed_files),
        ]
    )

    assert rc == 1


def test_main_fails_when_baseline_is_missing_repo_tool(tmp_path: Path) -> None:
    usr_tests = tmp_path / "src" / "dnadesign" / "usr" / "tests"
    infer_tests = tmp_path / "src" / "dnadesign" / "infer" / "tests"
    usr_tests.mkdir(parents=True, exist_ok=True)
    infer_tests.mkdir(parents=True, exist_ok=True)
    (usr_tests / "test_a.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")
    (infer_tests / "test_b.py").write_text("def test_b():\n    assert True\n", encoding="utf-8")

    baseline_path = tmp_path / "tool-coverage-baseline.json"
    baseline_path.write_text('{"usr": 0.0}', encoding="utf-8")

    rc = main(
        [
            "--event-name",
            "push",
            "--repo-root",
            str(tmp_path),
            "--baseline-json",
            str(baseline_path),
        ]
    )

    assert rc == 1


def test_main_fails_when_baseline_has_unknown_tool(tmp_path: Path) -> None:
    usr_tests = tmp_path / "src" / "dnadesign" / "usr" / "tests"
    usr_tests.mkdir(parents=True, exist_ok=True)
    (usr_tests / "test_a.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")

    baseline_path = tmp_path / "tool-coverage-baseline.json"
    baseline_path.write_text('{"usr": 0.0, "ghost": 0.0}', encoding="utf-8")

    rc = main(
        [
            "--event-name",
            "push",
            "--repo-root",
            str(tmp_path),
            "--baseline-json",
            str(baseline_path),
        ]
    )

    assert rc == 1


def test_main_fails_when_baseline_payload_is_not_object(tmp_path: Path) -> None:
    usr_tests = tmp_path / "src" / "dnadesign" / "usr" / "tests"
    usr_tests.mkdir(parents=True, exist_ok=True)
    (usr_tests / "test_a.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")

    baseline_path = tmp_path / "tool-coverage-baseline.json"
    baseline_path.write_text('["usr"]', encoding="utf-8")

    rc = main(
        [
            "--event-name",
            "push",
            "--repo-root",
            str(tmp_path),
            "--baseline-json",
            str(baseline_path),
        ]
    )

    assert rc == 1
