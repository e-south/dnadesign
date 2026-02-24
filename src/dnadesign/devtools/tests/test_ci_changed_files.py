"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/test_ci_changed_files.py

Tests for CI changed-file collection using real git repositories.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from dnadesign.devtools.ci_changed_files import collect_changed_files, main


def _run_git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _create_repo_with_feature_change(tmp_path: Path) -> tuple[Path, str]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    _run_git(repo_root, "init", "-b", "main")
    _run_git(repo_root, "config", "user.email", "test@example.com")
    _run_git(repo_root, "config", "user.name", "Test User")

    (repo_root / "README.md").write_text("base\n", encoding="utf-8")
    _run_git(repo_root, "add", "README.md")
    _run_git(repo_root, "commit", "-m", "base commit")

    _run_git(repo_root, "remote", "add", "origin", str(repo_root))
    _run_git(repo_root, "switch", "-c", "feature")
    (repo_root / "README.md").write_text("base\nfeature\n", encoding="utf-8")
    _run_git(repo_root, "add", "README.md")
    _run_git(repo_root, "commit", "-m", "feature commit")
    head_sha = _run_git(repo_root, "rev-parse", "HEAD")

    return repo_root, head_sha


def test_collect_changed_files_returns_empty_for_non_pr(tmp_path: Path) -> None:
    repo_root, head_sha = _create_repo_with_feature_change(tmp_path)

    files = collect_changed_files(
        event_name="push",
        repo_root=repo_root,
        base_ref="main",
        head_sha=head_sha,
    )

    assert files == []


def test_collect_changed_files_returns_pr_diff(tmp_path: Path) -> None:
    repo_root, head_sha = _create_repo_with_feature_change(tmp_path)

    files = collect_changed_files(
        event_name="pull_request",
        repo_root=repo_root,
        base_ref="main",
        head_sha=head_sha,
    )

    assert files == ["README.md"]


def test_collect_changed_files_uses_existing_tracking_ref_without_fetch(tmp_path: Path) -> None:
    repo_root, head_sha = _create_repo_with_feature_change(tmp_path)
    main_sha = _run_git(repo_root, "rev-parse", "main")
    _run_git(repo_root, "update-ref", "refs/remotes/broken/main", main_sha)

    files = collect_changed_files(
        event_name="pull_request",
        repo_root=repo_root,
        base_ref="main",
        head_sha=head_sha,
        remote="broken",
    )

    assert files == ["README.md"]


def test_main_fails_when_pr_args_missing(tmp_path: Path) -> None:
    repo_root, _ = _create_repo_with_feature_change(tmp_path)
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


def test_main_reports_git_fetch_failure_with_context(tmp_path: Path, capsys) -> None:
    repo_root = tmp_path / "repo-no-remote"
    repo_root.mkdir(parents=True, exist_ok=True)

    _run_git(repo_root, "init", "-b", "main")
    _run_git(repo_root, "config", "user.email", "test@example.com")
    _run_git(repo_root, "config", "user.name", "Test User")
    (repo_root / "README.md").write_text("base\n", encoding="utf-8")
    _run_git(repo_root, "add", "README.md")
    _run_git(repo_root, "commit", "-m", "base commit")
    head_sha = _run_git(repo_root, "rev-parse", "HEAD")

    output_file = tmp_path / "changed.txt"
    rc = main(
        [
            "--event-name",
            "pull_request",
            "--repo-root",
            str(repo_root),
            "--base-ref",
            "main",
            "--head-sha",
            head_sha,
            "--output-file",
            str(output_file),
        ]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert "git fetch failed" in captured.out


def test_main_fails_for_empty_remote_name(tmp_path: Path) -> None:
    repo_root, head_sha = _create_repo_with_feature_change(tmp_path)
    output_file = tmp_path / "changed.txt"

    rc = main(
        [
            "--event-name",
            "pull_request",
            "--repo-root",
            str(repo_root),
            "--base-ref",
            "main",
            "--head-sha",
            head_sha,
            "--remote",
            "",
            "--output-file",
            str(output_file),
        ]
    )

    assert rc == 1
