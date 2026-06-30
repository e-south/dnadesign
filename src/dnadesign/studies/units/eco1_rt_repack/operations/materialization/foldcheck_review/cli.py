"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_review/cli.py

CLI for Eco1 fold-check review materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review.pipeline import (
    materialize_foldcheck_review,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize Eco1 fold-check review ranking and panel artifacts.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--render-chimerax-overlay",
        action="store_true",
        help="Opt in to launching ChimeraX to render the selected-structure overlay PNG.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = materialize_foldcheck_review(
        repo_root=args.repo_root,
        output_root=args.output_root,
        render_chimerax_overlay=args.render_chimerax_overlay,
    )
    print(result.ranking_path)
    return 0
