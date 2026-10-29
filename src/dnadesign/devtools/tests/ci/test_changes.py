"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/ci/test_changes.py

Tests for CI change-scope detection used for core/external integration lane routing.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from dnadesign.devtools.ci.changes import (
    determine_scope,
    discover_external_integration_tools,
    discover_repo_tools,
    main,
)


@pytest.mark.parametrize(
    "path", ["README.md", "docs/dev/README.md", "src/dnadesign/opal/README.md", ".agents/skills/example/SKILL.md"]
)
def test_documentation_preparation_is_explicit(path: str) -> None:
    scope = determine_scope(
        event_name="pull_request",
        changed_files=[path],
        tool_names={"opal", "infer"},
        external_integration_tool_names={"infer"},
    )
    assert scope.documentation_only
    assert not scope.run_full_core


@pytest.mark.parametrize(
    "path",
    [
        ".github/tool-lifecycle.toml",
        "unknown.py",
        ".github/workflows/other.yaml",
        "src/dnadesign/unclassified/README.md",
    ],
)
def test_unclassified_and_lifecycle_changes_keep_broad_checks(path: str) -> None:
    scope = determine_scope(
        event_name="pull_request",
        changed_files=[path],
        tool_names={"opal", "infer"},
        external_integration_tool_names={"infer"},
    )
    assert scope.run_full_core
    assert not scope.documentation_only


@pytest.mark.parametrize(
    "path",
    [
        "src/dnadesign/opal/src/main.py",
        "src/dnadesign/opal/tests/fixtures/input.md",
        "src/dnadesign/opal/campaigns/demo/README.md",
    ],
)
def test_mixed_or_packaged_inputs_do_not_use_documentation_preparation(path: str) -> None:
    scope = determine_scope(
        event_name="pull_request",
        changed_files=["README.md", path],
        tool_names={"opal", "infer"},
        external_integration_tool_names={"infer"},
    )
    assert scope.affected_tools == ["opal"]
    assert not scope.documentation_only


def test_empty_pr_scope_cannot_select_documentation_preparation() -> None:
    with pytest.raises(ValueError, match="changed files"):
        determine_scope(
            event_name="pull_request", changed_files=[], tool_names={"opal"}, external_integration_tool_names=set()
        )


@pytest.mark.parametrize(
    ("provider", "consumer"), [("seqledger", "opal"), ("infer", "permuter"), ("baserender", "cruncher")]
)
def test_public_provider_changes_select_consumer_suites(provider: str, consumer: str) -> None:
    scope = determine_scope(
        event_name="pull_request",
        changed_files=[f"src/dnadesign/{provider}/__init__.py"],
        tool_names={provider, consumer, "junction"},
        external_integration_tool_names={consumer},
    )
    assert set(scope.affected_tools) == {provider, consumer}
    assert scope.external_integration_tools == [consumer]
    assert scope.run_external_integration


def test_consumer_scope_follows_transitive_edges_and_terminates_on_cycles() -> None:
    scope = determine_scope(
        event_name="pull_request",
        changed_files=["src/dnadesign/seqledger/__init__.py"],
        tool_names={"seqledger", "ops", "opal", "junction"},
        external_integration_tool_names=set(),
    )
    assert scope.affected_tools == ["opal", "ops", "seqledger"]


def test_non_pr_documentation_changes_still_run_full_validation() -> None:
    scope = determine_scope(
        event_name="push", changed_files=["README.md"], tool_names={"opal"}, external_integration_tool_names=set()
    )
    assert scope.run_full_core
    assert not scope.documentation_only


def test_executable_documentation_keeps_its_runtime_contract() -> None:
    scope = determine_scope(
        event_name="pull_request",
        changed_files=["src/dnadesign/opal/docs/workflows/seqledger-infer-x-active-learning.md"],
        tool_names={"opal", "seqledger"},
        external_integration_tool_names=set(),
    )
    assert scope.affected_tools == ["opal"]
    assert not scope.documentation_only


def test_determine_scope_non_pr_runs_full_core_and_external_integration() -> None:
    result = determine_scope(
        event_name="push",
        changed_files=[],
        tool_names={"infer", "seqledger"},
        external_integration_tool_names={"infer"},
    )

    assert result.run_full_core is True
    assert result.run_external_integration is True
    assert result.affected_tools == ["infer", "seqledger"]
    assert result.external_integration_tools == ["infer"]


def test_determine_scope_pr_selects_provider_and_declared_consumer() -> None:
    result = determine_scope(
        event_name="pull_request",
        changed_files=["src/dnadesign/seqledger/src/cli.py"],
        tool_names={"infer", "seqledger"},
        external_integration_tool_names={"infer"},
    )

    assert result.run_full_core is False
    assert result.run_external_integration is True
    assert result.affected_tools == ["infer", "seqledger"]
    assert result.external_integration_tools == ["infer"]


def test_determine_scope_pr_infer_change_triggers_external_integration_lane() -> None:
    result = determine_scope(
        event_name="pull_request",
        changed_files=["src/dnadesign/infer/src/cli/main.py"],
        tool_names={"infer", "seqledger"},
        external_integration_tool_names={"infer"},
    )

    assert result.run_full_core is False
    assert result.run_external_integration is True
    assert result.affected_tools == ["infer"]
    assert result.external_integration_tools == ["infer"]


@pytest.mark.parametrize("path", ["pyproject.toml", "uv.lock", "setup.py"])
def test_package_configuration_change_triggers_full_core_and_external_integration(path) -> None:
    result = determine_scope(
        event_name="pull_request",
        changed_files=[path],
        tool_names={"infer", "seqledger"},
        external_integration_tool_names={"infer"},
    )

    assert result.run_full_core is True
    assert result.run_external_integration is True
    assert result.affected_tools == ["infer", "seqledger"]
    assert result.external_integration_tools == ["infer"]


def test_determine_scope_pr_python_version_change_triggers_full_compatibility() -> None:
    result = determine_scope(
        event_name="pull_request",
        changed_files=[".python-version"],
        tool_names={"infer", "seqledger"},
        external_integration_tool_names={"infer"},
    )

    assert result.run_full_core is True
    assert result.run_external_integration is True
    assert result.affected_tools == ["infer", "seqledger"]
    assert result.external_integration_tools == ["infer"]


def test_determine_scope_pr_docs_only_skips_tool_coverage_gate() -> None:
    result = determine_scope(
        event_name="pull_request",
        changed_files=["docs/README.md"],
        tool_names={"infer", "seqledger"},
        external_integration_tool_names={"infer"},
    )

    assert result.run_full_core is False
    assert result.run_external_integration is False
    assert result.affected_tools == []
    assert result.external_integration_tools == []
    assert result.run_coverage_gate is False


@pytest.mark.parametrize("tool", ["infer", "opal"])
def test_tool_readme_edit_does_not_trigger_runtime_or_external_tests(tool) -> None:
    result = determine_scope(
        event_name="pull_request",
        changed_files=[f"src/dnadesign/{tool}/README.md"],
        tool_names={"infer", "opal"},
        external_integration_tool_names={"infer"},
    )
    assert not result.run_full_core
    assert not result.run_coverage_gate
    assert not result.run_external_integration


def test_determine_scope_pr_shared_code_change_triggers_full_core_and_external_integration() -> None:
    result = determine_scope(
        event_name="pull_request",
        changed_files=["src/dnadesign/devtools/quality/tool_coverage.py"],
        tool_names={"infer", "seqledger"},
        external_integration_tool_names={"infer"},
    )

    assert result.run_full_core is True
    assert result.run_external_integration is True
    assert result.affected_tools == ["infer", "seqledger"]
    assert result.external_integration_tools == ["infer"]


def test_determine_scope_pr_shared_package_root_change_triggers_full_core_and_external_integration() -> None:
    result = determine_scope(
        event_name="pull_request",
        changed_files=["src/dnadesign/__init__.py"],
        tool_names={"infer", "seqledger"},
        external_integration_tool_names={"infer"},
    )

    assert result.run_full_core is True
    assert result.run_external_integration is True
    assert result.affected_tools == ["infer", "seqledger"]
    assert result.external_integration_tools == ["infer"]


def test_determine_scope_pr_shared_contract_package_change_triggers_full_core() -> None:
    result = determine_scope(
        event_name="pull_request",
        changed_files=["src/dnadesign/contracts/artifacts.py"],
        tool_names={"contracts", "infer", "seqledger"},
        external_integration_tool_names={"infer"},
    )

    assert result.run_full_core is True
    assert result.run_external_integration is True
    assert result.affected_tools == ["contracts", "infer", "seqledger"]
    assert result.external_integration_tools == ["infer"]


def test_determine_scope_pr_external_integration_selection_includes_consumers() -> None:
    result = determine_scope(
        event_name="pull_request",
        changed_files=["src/dnadesign/cruncher/src/cli.py"],
        tool_names={"cruncher", "infer", "seqledger"},
        external_integration_tool_names={"cruncher", "infer"},
    )

    assert result.run_full_core is False
    assert result.run_external_integration is True
    assert result.affected_tools == ["cruncher", "infer", "seqledger"]
    assert result.external_integration_tools == ["cruncher", "infer"]


def test_determine_scope_pr_baseline_file_triggers_full_core() -> None:
    result = determine_scope(
        event_name="pull_request",
        changed_files=[".github/tool-coverage-baseline.json"],
        tool_names={"infer", "seqledger"},
        external_integration_tool_names={"infer"},
    )

    assert result.run_full_core is True
    assert result.run_external_integration is False
    assert result.affected_tools == ["infer", "seqledger"]
    assert result.external_integration_tools == ["infer"]


def test_determine_scope_disables_heavy_when_no_external_integration_tools_exist() -> None:
    result = determine_scope(
        event_name="push",
        changed_files=[],
        tool_names={"seqledger"},
        external_integration_tool_names=set(),
    )

    assert result.run_full_core is True
    assert result.run_external_integration is False
    assert result.affected_tools == ["seqledger"]
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
    (src_root / "seqledger").mkdir(parents=True, exist_ok=True)
    (src_root / "infer").mkdir(parents=True, exist_ok=True)
    (src_root / "devtools").mkdir(parents=True, exist_ok=True)
    (src_root / "__pycache__").mkdir(parents=True, exist_ok=True)
    (src_root / "archived").mkdir(parents=True, exist_ok=True)
    (src_root / "prototypes").mkdir(parents=True, exist_ok=True)

    tool_names = discover_repo_tools(repo_root=tmp_path)
    assert tool_names == {"infer", "seqledger"}


def test_discover_repo_tools_excludes_legacy_data_directories(tmp_path: Path) -> None:
    src_root = tmp_path / "src" / "dnadesign"
    (src_root / "seqledger").mkdir(parents=True, exist_ok=True)
    (src_root / "infer").mkdir(parents=True, exist_ok=True)
    (src_root / "densehairpins").mkdir(parents=True, exist_ok=True)
    (src_root / "evoinference").mkdir(parents=True, exist_ok=True)
    (src_root / "sequences").mkdir(parents=True, exist_ok=True)

    tool_names = discover_repo_tools(repo_root=tmp_path)
    assert tool_names == {"infer", "seqledger"}


def test_discover_repo_tools_keeps_latentdna_and_ignores_cache_only_dirs(tmp_path: Path) -> None:
    src_root = tmp_path / "src" / "dnadesign"
    (src_root / "latentdna").mkdir(parents=True, exist_ok=True)
    (src_root / "latentdna" / "__init__.py").write_text("", encoding="utf-8")
    (src_root / "stale_cache" / "__pycache__").mkdir(parents=True, exist_ok=True)

    tool_names = discover_repo_tools(repo_root=tmp_path)
    assert tool_names == {"latentdna"}


@pytest.mark.parametrize("link_name", ["__pycache__", "source-link"])
def test_discover_repo_tools_keeps_symlinks_without_following_them(tmp_path: Path, link_name: str) -> None:
    src_root = tmp_path / "src" / "dnadesign"
    external = tmp_path / "external-cache"
    external.mkdir(parents=True)
    candidate = src_root / "stale_cache"
    candidate.mkdir(parents=True)
    (candidate / link_name).symlink_to(external, target_is_directory=True)
    (external / "loop").symlink_to(external, target_is_directory=True)

    assert discover_repo_tools(repo_root=tmp_path) == {"stale_cache"}


def test_discover_repo_tools_ignores_nested_cache_only_dirs(tmp_path: Path) -> None:
    src_root = tmp_path / "src" / "dnadesign"
    (src_root / "latentdna").mkdir(parents=True, exist_ok=True)
    (src_root / "latentdna" / "__init__.py").write_text("", encoding="utf-8")
    cache_root = src_root / "retired_tool" / "compiler" / "contracts" / "__pycache__"
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "models.cpython-312.pyc").write_bytes(b"cached")

    tool_names = discover_repo_tools(repo_root=tmp_path)

    assert tool_names == {"latentdna"}


def test_discover_repo_tools_ignores_untracked_local_backup(tmp_path: Path) -> None:
    src_root = tmp_path / "src" / "dnadesign"
    (src_root / "latentdna").mkdir(parents=True, exist_ok=True)
    (src_root / "latentdna" / "__init__.py").write_text("", encoding="utf-8")
    (src_root / "studies").mkdir(parents=True, exist_ok=True)
    (src_root / "studies" / "private.py").write_text("SECRET = True\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("src/dnadesign/studies/\n", encoding="utf-8")
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)

    tool_names = discover_repo_tools(repo_root=tmp_path)

    assert tool_names == {"latentdna"}


def test_discover_repo_tools_keeps_tracked_ignored_package(tmp_path: Path) -> None:
    src_root = tmp_path / "src" / "dnadesign"
    (src_root / "studies").mkdir(parents=True, exist_ok=True)
    (src_root / "studies" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("src/dnadesign/studies/\n", encoding="utf-8")
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "add", "--force", "src/dnadesign/studies/__init__.py"],
        cwd=tmp_path,
        check=True,
    )

    tool_names = discover_repo_tools(repo_root=tmp_path)

    assert tool_names == {"studies"}


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
    src_root = tmp_path / "src" / "dnadesign" / "seqledger" / "tests"
    src_root.mkdir(parents=True, exist_ok=True)
    (src_root / "test_a.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")

    baseline_path = tmp_path / "tool-coverage-baseline.json"
    baseline_path.write_text('{"seqledger": 0.0}', encoding="utf-8")

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
    usr_tests = tmp_path / "src" / "dnadesign" / "seqledger" / "tests"
    usr_tests.mkdir(parents=True, exist_ok=True)
    (usr_tests / "test_a.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")

    baseline_path = tmp_path / "tool-coverage-baseline.json"
    baseline_path.write_text('{"seqledger": 0.0}', encoding="utf-8")

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
    usr_tests = tmp_path / "src" / "dnadesign" / "seqledger" / "tests"
    infer_tests = tmp_path / "src" / "dnadesign" / "infer" / "tests"
    usr_tests.mkdir(parents=True, exist_ok=True)
    infer_tests.mkdir(parents=True, exist_ok=True)
    (usr_tests / "test_a.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")
    (infer_tests / "test_b.py").write_text("def test_b():\n    assert True\n", encoding="utf-8")

    baseline_path = tmp_path / "tool-coverage-baseline.json"
    baseline_path.write_text('{"seqledger": 0.0}', encoding="utf-8")

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
    usr_tests = tmp_path / "src" / "dnadesign" / "seqledger" / "tests"
    usr_tests.mkdir(parents=True, exist_ok=True)
    (usr_tests / "test_a.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")

    baseline_path = tmp_path / "tool-coverage-baseline.json"
    baseline_path.write_text('{"seqledger": 0.0, "ghost": 0.0}', encoding="utf-8")

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
    usr_tests = tmp_path / "src" / "dnadesign" / "seqledger" / "tests"
    usr_tests.mkdir(parents=True, exist_ok=True)
    (usr_tests / "test_a.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")

    baseline_path = tmp_path / "tool-coverage-baseline.json"
    baseline_path.write_text('["seqledger"]', encoding="utf-8")

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
