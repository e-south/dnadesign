"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/thread_plan/cli.py

CLI for Eco1 RT thread-plan materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.thread_plan.pipeline import (
    _DEFAULT_CREATED_AT,
    _DEFAULT_OUTPUT_ROOT,
    materialize_thread_plan,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the thread-plan CLI parser."""

    parser = argparse.ArgumentParser(
        description="Materialize Eco1 RT thread_plan.yaml from the accepted mask_set.yaml."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--created-at", default=_DEFAULT_CREATED_AT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run thread-plan materialization and print emitted paths."""

    args = build_parser().parse_args(argv)
    result = materialize_thread_plan(
        repo_root=args.repo_root,
        output_root=args.output_root,
        created_at=args.created_at,
    )
    print(json.dumps({"thread_plan_path": str(result.thread_plan_path)}, indent=2, sort_keys=True))
    return 0
