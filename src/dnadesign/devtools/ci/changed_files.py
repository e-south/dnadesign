"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/ci/changed_files.py

Collects changed file paths for CI scope detection in pull request workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run_git(*, repo_root: Path, args: list[str], context: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        detail = f" ({stderr})" if stderr else ""
        raise ValueError(f"{context}: git {' '.join(args)} failed{detail}") from exc
    return completed.stdout


def _resolve_commit(*, repo_root: Path, commit: str, role: str) -> str:
    try:
        return _run_git(
            repo_root=repo_root,
            args=["rev-parse", "--verify", f"{commit}^{{commit}}"],
            context=f"{role} commit is unavailable in the CI checkout",
        ).strip()
    except ValueError as exc:
        raise ValueError(f"{role} commit is unavailable in the CI checkout: {commit}") from exc


def collect_changed_files(*, event_name: str, repo_root: Path, base_sha: str | None, head_sha: str | None) -> list[str]:
    if event_name != "pull_request":
        return []

    if not base_sha or not head_sha:
        raise ValueError("--base-sha and --head-sha are required for pull_request event.")

    resolved_base = _resolve_commit(repo_root=repo_root, commit=base_sha, role="pull-request base")
    resolved_head = _resolve_commit(repo_root=repo_root, commit=head_sha, role="pull-request merge")
    parent_line = _run_git(
        repo_root=repo_root,
        args=["rev-list", "--parents", "-n", "1", resolved_head],
        context="unable to inspect pull-request merge parents",
    ).strip()
    parent_tokens = parent_line.split()
    if len(parent_tokens) != 3:
        raise ValueError("--head-sha must resolve to GitHub's two-parent pull-request merge commit.")
    if parent_tokens[1] != resolved_base:
        raise ValueError("pull-request merge commit first parent does not match --base-sha.")

    diff_output = _run_git(
        repo_root=repo_root,
        args=["diff", "--name-only", f"{resolved_base}...{resolved_head}"],
        context="git diff failed",
    )
    return [line.strip() for line in diff_output.splitlines() if line.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect changed files for CI scope detection.")
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--base-sha", default=None)
    parser.add_argument("--head-sha", default=None)
    parser.add_argument("--output-file", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        changed_files = collect_changed_files(
            event_name=args.event_name,
            repo_root=args.repo_root,
            base_sha=args.base_sha,
            head_sha=args.head_sha,
        )
    except ValueError as exc:
        print(str(exc))
        return 1

    args.output_file.write_text(
        "".join(f"{path}\n" for path in changed_files),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
