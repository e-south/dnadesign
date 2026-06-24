"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/manual_mask_authority/cli.py

CLI for Eco1 RT manual mask-authority materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.manual_mask_authority.pipeline import (
    _DEFAULT_CREATED_AT,
    _DEFAULT_OUTPUT_ROOT,
    materialize_manual_mask_authority,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the manual mask-authority CLI parser."""

    parser = argparse.ArgumentParser(description="Materialize Eco1 RT manual_mask_authority.yaml.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--created-at", default=_DEFAULT_CREATED_AT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run manual mask-authority materialization and print emitted paths."""

    args = build_parser().parse_args(argv)
    result = materialize_manual_mask_authority(
        repo_root=args.repo_root,
        output_root=args.output_root,
        created_at=args.created_at,
    )
    print(json.dumps({"manual_mask_authority_path": str(result.manual_mask_authority_path)}, indent=2, sort_keys=True))
    return 0
