"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/ci_changed_files.py

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


def collect_changed_files(
    *, event_name: str, repo_root: Path, base_ref: str | None, head_sha: str | None, remote: str = "origin"
) -> list[str]:
    if event_name != "pull_request":
        return []

    if not base_ref or not head_sha:
        raise ValueError("--base-ref and --head-sha are required for pull_request event.")
    if not remote.strip():
        raise ValueError("--remote must not be empty.")

    _run_git(
        repo_root=repo_root,
        args=["fetch", "--no-tags", "--depth=1", remote, base_ref],
        context="git fetch failed",
    )
    diff_output = _run_git(
        repo_root=repo_root,
        args=["diff", "--name-only", f"{remote}/{base_ref}...{head_sha}"],
        context="git diff failed",
    )
    return [line.strip() for line in diff_output.splitlines() if line.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect changed files for CI scope detection.")
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--base-ref", default=None)
    parser.add_argument("--head-sha", default=None)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--output-file", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        changed_files = collect_changed_files(
            event_name=args.event_name,
            repo_root=args.repo_root,
            base_ref=args.base_ref,
            head_sha=args.head_sha,
            remote=args.remote,
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
