"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/structure_preprocessing/cli.py

CLI for Eco1 RT structure-preprocessing provenance materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.structure_preprocessing.pipeline import (
    _DEFAULT_CREATED_AT,
    _DEFAULT_OUTPUT_ROOT,
    materialize_structure_preprocessing_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the structure-preprocessing CLI parser."""

    parser = argparse.ArgumentParser(
        description="Materialize Eco1 RT structure_preprocessing_manifest.yaml from selected 7V9U provenance."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--created-at", default=_DEFAULT_CREATED_AT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run structure-preprocessing materialization and print emitted paths."""

    args = build_parser().parse_args(argv)
    result = materialize_structure_preprocessing_manifest(
        repo_root=args.repo_root,
        output_root=args.output_root,
        created_at=args.created_at,
    )
    print(
        json.dumps(
            {"structure_preprocessing_manifest_path": str(result.structure_preprocessing_manifest_path)},
            indent=2,
            sort_keys=True,
        )
    )
    return 0
