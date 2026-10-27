"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/docs/freshness.py

Change-aware verification dates for documentation contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import datetime as dt
import subprocess
from collections.abc import Mapping
from pathlib import Path, PurePosixPath


def collect_changed_doc_dates(
    repo_root: Path,
    *,
    changed_files_file: Path | None = None,
) -> dict[str, dt.date]:
    """Return changed Markdown paths and the stable date of each change event."""

    root = repo_root.expanduser().resolve()
    explicit_paths = _load_explicit_changed_paths(root, changed_files_file)
    dirty_paths = _git_dirty_paths(root)
    changed_paths = explicit_paths | dirty_paths
    clean_markdown_paths = sorted(
        relative_path
        for relative_path in changed_paths - dirty_paths
        if relative_path.endswith(".md") and (root / relative_path).is_file()
    )
    committed_change_dates = _git_last_change_dates(root, clean_markdown_paths)
    result: dict[str, dt.date] = {}
    for relative_path in sorted(changed_paths):
        if not relative_path.endswith(".md"):
            continue
        path = root / relative_path
        if not path.is_file():
            continue
        if relative_path in dirty_paths:
            result[relative_path] = dt.datetime.fromtimestamp(path.stat().st_mtime).date()
            continue
        result[relative_path] = committed_change_dates.get(
            relative_path,
            dt.datetime.fromtimestamp(path.stat().st_mtime).date(),
        )
    return result


def verification_change_issue(
    *,
    repo_root: Path,
    path: Path,
    last_verified: dt.date,
    changed_doc_dates: Mapping[str, dt.date],
) -> str | None:
    """Reject verification metadata that predates a known document change."""

    root = repo_root.expanduser().resolve()
    try:
        relative_path = path.expanduser().resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"documentation path escapes repository root: {path}") from exc
    changed_on = changed_doc_dates.get(relative_path)
    if changed_on is None or last_verified >= changed_on:
        return None
    return (
        f"{path}: Last verified date {last_verified.isoformat()} predates this document's "
        f"{changed_on.isoformat()} change; review the changed content and update the date."
    )


def _load_explicit_changed_paths(root: Path, changed_files_file: Path | None) -> set[str]:
    if changed_files_file is None:
        return set()
    source = changed_files_file.expanduser()
    if not source.is_absolute():
        source = root / source
    if not source.is_file():
        raise FileNotFoundError(f"changed-files input is missing: {source}")
    return {
        _normalize_repo_relative_path(raw_path)
        for raw_path in source.read_text(encoding="utf-8").splitlines()
        if raw_path.strip()
    }


def _normalize_repo_relative_path(raw_path: str) -> str:
    value = raw_path.strip().replace("\\", "/")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise ValueError(f"changed-files path must be repository-relative: {raw_path!r}")
    return candidate.as_posix()


def _git_dirty_paths(root: Path) -> set[str]:
    commands = (
        ("git", "diff", "--name-only", "--diff-filter=ACMR", "-z"),
        ("git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"),
        ("git", "ls-files", "--others", "--exclude-standard", "-z"),
    )
    paths: set[str] = set()
    for command in commands:
        completed = subprocess.run(command, cwd=root, check=False, capture_output=True)
        if completed.returncode != 0:
            return set()
        paths.update(
            _normalize_repo_relative_path(item.decode("utf-8")) for item in completed.stdout.split(b"\0") if item
        )
    return paths


def _git_last_change_dates(root: Path, relative_paths: list[str]) -> dict[str, dt.date]:
    if not relative_paths:
        return {}
    completed = subprocess.run(
        ("git", "log", "--format=%x1e%cs", "--name-only", "--", *relative_paths),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {}

    requested = set(relative_paths)
    change_dates: dict[str, dt.date] = {}
    for record in completed.stdout.split("\x1e"):
        lines = [line.strip() for line in record.splitlines() if line.strip()]
        if not lines:
            continue
        value = lines[0]
        try:
            changed_on = dt.date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"git returned a non-ISO change date: {value!r}") from exc
        for relative_path in lines[1:]:
            if relative_path in requested:
                change_dates.setdefault(relative_path, changed_on)
    return change_dates


__all__ = ["collect_changed_doc_dates", "verification_change_issue"]
