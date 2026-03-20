"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/generate_runbook_catalog.py

Regenerates the shared runbook catalog procedure and tool-source sections from
owner-local metadata sidecars.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dnadesign.ops.catalog import rewrite_runbook_catalog_sections


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate docs/runbooks/README.md procedure and tool-source rows from owner-local metadata sidecars."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root containing docs/runbooks/README.md.",
    )
    args = parser.parse_args(argv)

    updated_path = rewrite_runbook_catalog_sections(repo_root=args.repo_root)
    print(f"Updated {updated_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
