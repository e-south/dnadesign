"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/docs/test_freshness.py

Tests for change-aware documentation verification dates.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import datetime as dt
import os
import subprocess
from pathlib import Path

import pytest

from dnadesign.devtools.docs.freshness import collect_changed_doc_dates, verification_change_issue


def _git(repo_root: Path, *args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(["git", *args], cwd=repo_root, check=True, capture_output=True, text=True, env=env)


def _init_repo(repo_root: Path) -> None:
    _git(repo_root, "init")
    _git(repo_root, "config", "user.email", "docs@example.test")
    _git(repo_root, "config", "user.name", "Docs Test")


def test_unchanged_document_does_not_expire_with_age(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    path = tmp_path / "docs" / "guide.md"
    path.parent.mkdir()
    path.write_text("## Guide\n\n**Last verified:** 2020-01-01\n", encoding="utf-8")
    _git(tmp_path, "add", "docs/guide.md")
    commit_env = {
        **os.environ,
        "GIT_AUTHOR_DATE": "2020-01-01T12:00:00+00:00",
        "GIT_COMMITTER_DATE": "2020-01-01T12:00:00+00:00",
    }
    _git(tmp_path, "commit", "-m", "add guide", env=commit_env)

    assert collect_changed_doc_dates(tmp_path) == {}


def test_dirty_document_uses_its_change_date(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    path = tmp_path / "docs" / "guide.md"
    path.parent.mkdir()
    path.write_text("## Guide\n", encoding="utf-8")
    _git(tmp_path, "add", "docs/guide.md")
    _git(tmp_path, "commit", "-m", "add guide")
    path.write_text("## Changed guide\n", encoding="utf-8")

    changed = collect_changed_doc_dates(tmp_path)

    assert changed == {"docs/guide.md": dt.datetime.fromtimestamp(path.stat().st_mtime).date()}


def test_explicit_committed_changes_use_one_git_history_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_repo(tmp_path)
    guide_path = tmp_path / "docs" / "guide.md"
    guide_path.parent.mkdir()
    guide_path.write_text("## Guide\n", encoding="utf-8")
    _git(tmp_path, "add", "docs/guide.md")
    guide_commit_env = {
        **os.environ,
        "GIT_AUTHOR_DATE": "2024-02-03T12:00:00+00:00",
        "GIT_COMMITTER_DATE": "2024-02-03T12:00:00+00:00",
    }
    _git(tmp_path, "commit", "-m", "add guide", env=guide_commit_env)
    reference_path = tmp_path / "docs" / "reference.md"
    reference_path.write_text("## Reference\n", encoding="utf-8")
    _git(tmp_path, "add", "docs/reference.md")
    reference_commit_env = {
        **os.environ,
        "GIT_AUTHOR_DATE": "2024-03-04T12:00:00+00:00",
        "GIT_COMMITTER_DATE": "2024-03-04T12:00:00+00:00",
    }
    _git(tmp_path, "commit", "-m", "add reference", env=reference_commit_env)
    changed_files = tmp_path / "changed.txt"
    changed_files.write_text("docs/guide.md\ndocs/reference.md\n", encoding="utf-8")
    git_log_calls: list[tuple[str, ...]] = []
    real_run = subprocess.run

    def _recording_run(command, *args, **kwargs):
        if tuple(command[:2]) == ("git", "log"):
            git_log_calls.append(tuple(command))
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr("dnadesign.devtools.docs.freshness.subprocess.run", _recording_run)

    changed = collect_changed_doc_dates(tmp_path, changed_files_file=changed_files)

    assert changed == {
        "docs/guide.md": dt.date(2024, 2, 3),
        "docs/reference.md": dt.date(2024, 3, 4),
    }
    assert len(git_log_calls) == 1
    assert set(git_log_calls[0][-2:]) == {"docs/guide.md", "docs/reference.md"}


def test_verification_must_cover_known_change(tmp_path: Path) -> None:
    path = tmp_path / "docs" / "guide.md"
    path.parent.mkdir()
    path.write_text("## Guide\n", encoding="utf-8")
    changed = {"docs/guide.md": dt.date(2026, 7, 12)}

    issue = verification_change_issue(
        repo_root=tmp_path,
        path=path,
        last_verified=dt.date(2026, 7, 11),
        changed_doc_dates=changed,
    )

    assert issue is not None
    assert "predates this document's 2026-07-12 change" in issue
    assert (
        verification_change_issue(
            repo_root=tmp_path,
            path=path,
            last_verified=dt.date(2026, 7, 12),
            changed_doc_dates=changed,
        )
        is None
    )


def test_changed_file_paths_must_be_repo_relative(tmp_path: Path) -> None:
    changed_files = tmp_path / "changed.txt"
    changed_files.write_text("../outside.md\n", encoding="utf-8")

    with pytest.raises(ValueError, match="repository-relative"):
        collect_changed_doc_dates(tmp_path, changed_files_file=changed_files)
