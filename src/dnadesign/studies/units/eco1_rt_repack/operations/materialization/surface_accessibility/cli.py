"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/surface_accessibility/cli.py

CLI for Eco1 RT surface-accessibility materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.surface_accessibility.pipeline import (
    materialize_surface_accessibility_profile,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""

    parser = argparse.ArgumentParser(description="Materialize Eco1 RT surface-accessibility evidence.")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = build_parser().parse_args(argv)
    result = materialize_surface_accessibility_profile(repo_root=args.repo_root, output_root=args.output_root)
    print(result.surface_accessibility_profile_path)
    return 0
