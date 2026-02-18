"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/meme_env.py

Resolves and exports the project-local MEME/FIMO tool path for CI and local use.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path


def resolve_meme_bin(*, repo_root: Path) -> Path:
    meme_bin = repo_root / ".pixi" / "envs" / "default" / "bin"
    fimo_path = meme_bin / "fimo"
    if not fimo_path.is_file():
        raise ValueError(f"FIMO binary not found at {fimo_path}. Run `pixi install --locked` first.")
    if not os.access(fimo_path, os.X_OK):
        raise ValueError(f"FIMO binary is not executable: {fimo_path}")
    return meme_bin


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve project-local MEME/FIMO runtime paths.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--github-env", type=Path, default=None)
    parser.add_argument("--github-path", type=Path, default=None)
    parser.add_argument(
        "--print-shell-export",
        action="store_true",
        help="Print shell export statements for MEME_BIN and PATH.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        meme_bin = resolve_meme_bin(repo_root=args.repo_root)
    except ValueError as exc:
        print(str(exc))
        return 1

    if args.github_env is not None:
        _append_line(args.github_env, f"MEME_BIN={meme_bin}")
    if args.github_path is not None:
        _append_line(args.github_path, str(meme_bin))
    if args.print_shell_export:
        print(f"export MEME_BIN={shlex.quote(str(meme_bin))}")
        print('export PATH="$MEME_BIN:$PATH"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
