"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/ci/test_changed_files.py

Tests for CI changed-file collection using real git repositories.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from dnadesign.devtools.ci.changed_files import collect_changed_files, main


def _run_git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _create_repo_with_pr_merge(tmp_path: Path) -> tuple[Path, str, str, str]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    _run_git(repo_root, "init", "-b", "main")
    _run_git(repo_root, "config", "user.email", "test@example.com")
    _run_git(repo_root, "config", "user.name", "Test User")

    (repo_root / "README.md").write_text("base\n", encoding="utf-8")
    _run_git(repo_root, "add", "README.md")
    _run_git(repo_root, "commit", "-m", "base commit")
    base_sha = _run_git(repo_root, "rev-parse", "HEAD")

    _run_git(repo_root, "switch", "-c", "feature")
    (repo_root / "README.md").write_text("base\nfeature\n", encoding="utf-8")
    _run_git(repo_root, "add", "README.md")
    _run_git(repo_root, "commit", "-m", "feature commit")
    feature_sha = _run_git(repo_root, "rev-parse", "HEAD")

    _run_git(repo_root, "switch", "main")
    _run_git(repo_root, "merge", "--no-ff", "--no-edit", feature_sha)
    merge_sha = _run_git(repo_root, "rev-parse", "HEAD")
    return repo_root, base_sha, feature_sha, merge_sha


def test_collect_changed_files_returns_empty_for_non_pr(tmp_path: Path) -> None:
    repo_root, _, _, _ = _create_repo_with_pr_merge(tmp_path)

    files = collect_changed_files(
        event_name="push",
        repo_root=repo_root,
        base_sha=None,
        head_sha=None,
    )

    assert files == []


def test_collect_changed_files_returns_pr_diff(tmp_path: Path) -> None:
    repo_root, base_sha, _, merge_sha = _create_repo_with_pr_merge(tmp_path)

    files = collect_changed_files(
        event_name="pull_request",
        repo_root=repo_root,
        base_sha=base_sha,
        head_sha=merge_sha,
    )

    assert files == ["README.md"]


def test_collect_changed_files_rejects_non_merge_head(tmp_path: Path) -> None:
    repo_root, base_sha, feature_sha, _ = _create_repo_with_pr_merge(tmp_path)

    try:
        collect_changed_files(
            event_name="pull_request",
            repo_root=repo_root,
            base_sha=base_sha,
            head_sha=feature_sha,
        )
    except ValueError as exc:
        assert "two-parent pull-request merge commit" in str(exc)
    else:
        raise AssertionError("non-merge pull-request head was accepted")


def test_collect_changed_files_rejects_base_snapshot_mismatch(tmp_path: Path) -> None:
    repo_root, _, feature_sha, merge_sha = _create_repo_with_pr_merge(tmp_path)

    try:
        collect_changed_files(
            event_name="pull_request",
            repo_root=repo_root,
            base_sha=feature_sha,
            head_sha=merge_sha,
        )
    except ValueError as exc:
        assert "first parent does not match --base-sha" in str(exc)
    else:
        raise AssertionError("mismatched pull-request base snapshot was accepted")


def test_collect_changed_files_ignores_later_base_branch_movement(tmp_path: Path) -> None:
    repo_root, base_sha, _, merge_sha = _create_repo_with_pr_merge(tmp_path)
    (repo_root / "later.txt").write_text("later base movement\n", encoding="utf-8")
    _run_git(repo_root, "add", "later.txt")
    _run_git(repo_root, "commit", "-m", "later base movement")

    files = collect_changed_files(
        event_name="pull_request",
        repo_root=repo_root,
        base_sha=base_sha,
        head_sha=merge_sha,
    )

    assert files == ["README.md"]


def test_main_fails_when_pr_args_missing(tmp_path: Path) -> None:
    repo_root, _, _, _ = _create_repo_with_pr_merge(tmp_path)
    output_file = tmp_path / "changed.txt"

    rc = main(
        [
            "--event-name",
            "pull_request",
            "--repo-root",
            str(repo_root),
            "--output-file",
            str(output_file),
        ]
    )

    assert rc == 1


def test_main_reports_unavailable_base_snapshot(tmp_path: Path, capsys) -> None:
    repo_root, _, _, merge_sha = _create_repo_with_pr_merge(tmp_path)

    output_file = tmp_path / "changed.txt"
    rc = main(
        [
            "--event-name",
            "pull_request",
            "--repo-root",
            str(repo_root),
            "--base-sha",
            "0" * 40,
            "--head-sha",
            merge_sha,
            "--output-file",
            str(output_file),
        ]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert "pull-request base commit is unavailable" in captured.out
