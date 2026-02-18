"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/test_ci_test_targets.py

Tests for CI helper that resolves affected tool test directories.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.devtools.ci_test_targets import main, resolve_test_targets


def test_resolve_test_targets_returns_existing_test_dirs(tmp_path: Path) -> None:
    usr_tests = tmp_path / "src" / "dnadesign" / "usr" / "tests"
    dense_tests = tmp_path / "src" / "dnadesign" / "densegen" / "tests"
    usr_tests.mkdir(parents=True, exist_ok=True)
    dense_tests.mkdir(parents=True, exist_ok=True)

    targets = resolve_test_targets(repo_root=tmp_path, tool_names=["usr", "densegen"])

    assert targets == [str(usr_tests), str(dense_tests)]


def test_resolve_test_targets_skips_tools_without_tests(tmp_path: Path) -> None:
    (tmp_path / "src" / "dnadesign" / "usr").mkdir(parents=True, exist_ok=True)

    targets = resolve_test_targets(repo_root=tmp_path, tool_names=["usr"])

    assert targets == []


def test_main_fails_for_unknown_tool(tmp_path: Path) -> None:
    (tmp_path / "src" / "dnadesign" / "usr" / "tests").mkdir(parents=True, exist_ok=True)

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
    (tmp_path / "src" / "dnadesign" / "usr" / "tests").mkdir(parents=True, exist_ok=True)

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
    usr_tests = tmp_path / "src" / "dnadesign" / "usr" / "tests"
    usr_tests.mkdir(parents=True, exist_ok=True)

    rc = main(
        [
            "--repo-root",
            str(tmp_path),
            "--affected-tools-csv",
            "usr",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == f"{usr_tests}\n"
