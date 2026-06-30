"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/cli.py

CLI for Eco1 review-deliverable materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.pipeline import (
    materialize_review_deliverables,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Materialize Eco1 review-deliverable visual bundle.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root.")
    parser.add_argument("--output-root", type=Path, default=None, help="Override study thread output root.")
    parser.add_argument(
        "--render-chimerax-png",
        action="store_true",
        help="Opt in to launching ChimeraX to render the optional mask-context PNG.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Eco1 review-deliverable materializer."""

    args = build_parser().parse_args(argv)
    result = materialize_review_deliverables(
        repo_root=args.repo_root,
        output_root=args.output_root,
        render_chimerax_png=args.render_chimerax_png,
    )
    print(f"review_deliverable_manifest: {result.manifest_path}")
    print(f"review_deliverable_notebook: {result.notebook_path}")
    print(f"deliverable_count: {result.deliverable_count}")
    return 0
