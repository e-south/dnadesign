"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/docs/banners/__main__.py

Provides the command-line entry point for banner generation and drift checks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .render import check_banners, render_banners


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate dnadesign documentation banners.")
    parser.add_argument("--check", action="store_true", help="Fail when a banner differs from its source.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    if args.check:
        stale = check_banners(args.repo_root)
        if stale:
            for path in stale:
                print(path)
            return 1
        return 0

    for path in render_banners(args.repo_root):
        print(path)
    return 0


raise SystemExit(main())
