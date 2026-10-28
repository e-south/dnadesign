"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/ci/test_test_targets.py

Tests for CI helper that resolves affected tool test directories.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from dnadesign.devtools.ci.changes import determine_scope
from dnadesign.devtools.ci.test_targets import main, resolve_test_targets


def test_provider_break_is_detected_by_selected_consumer_even_with_consumer_docs_edit(tmp_path: Path) -> None:
    provider = tmp_path / "src/dnadesign/seqledger/__init__.py"
    provider.parent.mkdir(parents=True)
    provider.write_text("def require_root():\n    return None\n")
    consumer = tmp_path / "src/dnadesign/opal/tests/test_contract.py"
    consumer.parent.mkdir(parents=True)
    consumer.write_text(
        "import importlib.util\nfrom pathlib import Path\n"
        "def test_requires_explicit_root():\n"
        "    path = Path(__file__).parents[2] / 'seqledger' / '__init__.py'\n"
        "    spec = importlib.util.spec_from_file_location('provider', path)\n"
        "    provider = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(provider)\n"
        "    assert provider.require_root() is not None\n"
    )
    changed = ["src/dnadesign/seqledger/__init__.py", "src/dnadesign/opal/README.md"]
    scope = determine_scope(
        event_name="pull_request",
        changed_files=changed,
        tool_names={"seqledger", "opal"},
        external_integration_tool_names=set(),
    )
    targets = resolve_test_targets(repo_root=tmp_path, tool_names=scope.affected_tools, changed_files=changed)
    assert str(consumer.parent) in targets
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *targets], cwd=tmp_path, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 1
    assert "test_requires_explicit_root" in result.stdout
    assert "assert None is not None" in result.stdout


def test_resolve_test_targets_returns_existing_test_dirs(tmp_path: Path) -> None:
    usr_tests = tmp_path / "src" / "dnadesign" / "seqledger" / "tests"
    dense_tests = tmp_path / "src" / "dnadesign" / "densegen" / "tests"
    usr_tests.mkdir(parents=True, exist_ok=True)
    dense_tests.mkdir(parents=True, exist_ok=True)

    targets = resolve_test_targets(repo_root=tmp_path, tool_names=["seqledger", "densegen"])

    assert targets == [str(usr_tests), str(dense_tests)]


def test_resolve_test_targets_skips_tools_without_tests(tmp_path: Path) -> None:
    (tmp_path / "src" / "dnadesign" / "seqledger").mkdir(parents=True, exist_ok=True)

    targets = resolve_test_targets(repo_root=tmp_path, tool_names=["seqledger"])

    assert targets == []


def test_docs_only_targets_keep_example_checks_without_full_runtime_suite(tmp_path: Path) -> None:
    tests = tmp_path / "src/dnadesign/opal/tests/cli"
    tests.mkdir(parents=True)
    (tests / "test_docs_workflow.py").touch()
    (tests / "test_runtime.py").touch()
    targets = resolve_test_targets(
        repo_root=tmp_path, tool_names=["opal"], changed_files=["src/dnadesign/opal/README.md"]
    )
    assert targets == [str(tests / "test_docs_workflow.py")]


def test_resolve_test_targets_includes_cluster_owned_cli_tests(tmp_path: Path) -> None:
    cluster_tests = tmp_path / "src" / "dnadesign" / "cluster" / "tests"
    cluster_cli_tests = tmp_path / "src" / "dnadesign" / "cluster" / "src" / "cli" / "tests"
    cluster_tests.mkdir(parents=True, exist_ok=True)
    cluster_cli_tests.mkdir(parents=True, exist_ok=True)

    targets = resolve_test_targets(repo_root=tmp_path, tool_names=["cluster"])

    assert targets == [str(cluster_tests), str(cluster_cli_tests)]


def test_resolve_test_targets_includes_external_study_fixture_guard_for_notify_and_ops(tmp_path: Path) -> None:
    guard = tmp_path / "src" / "dnadesign" / "devtools" / "tests" / "docs" / "test_external_study_boundaries.py"
    guard.parent.mkdir(parents=True)
    guard.touch()

    for tool_name in ("notify", "ops"):
        tool_tests = tmp_path / "src" / "dnadesign" / tool_name / "tests"
        tool_tests.mkdir(parents=True)

        targets = resolve_test_targets(
            repo_root=tmp_path,
            tool_names=[tool_name],
            changed_files=[f"src/dnadesign/{tool_name}/tests/test_fixture.py"],
        )

        assert targets == [str(tool_tests), str(guard)]


def test_main_fails_for_unknown_tool(tmp_path: Path) -> None:
    (tmp_path / "src" / "dnadesign" / "seqledger" / "tests").mkdir(parents=True, exist_ok=True)

    rc = main(
        [
            "--repo-root",
            str(tmp_path),
            "--affected-tools-csv",
            "usr,ghost",
        ]
    )

    assert rc == 1


def test_main_fails_for_empty_tool_list(tmp_path: Path) -> None:
    (tmp_path / "src" / "dnadesign" / "seqledger" / "tests").mkdir(parents=True, exist_ok=True)

    rc = main(
        [
            "--repo-root",
            str(tmp_path),
            "--affected-tools-csv",
            "",
        ]
    )

    assert rc == 1


def test_main_prints_one_target_per_line(tmp_path: Path, capsys) -> None:
    usr_tests = tmp_path / "src" / "dnadesign" / "seqledger" / "tests"
    usr_tests.mkdir(parents=True, exist_ok=True)

    rc = main(
        [
            "--repo-root",
            str(tmp_path),
            "--affected-tools-csv",
            "seqledger",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == f"{usr_tests}\n"
