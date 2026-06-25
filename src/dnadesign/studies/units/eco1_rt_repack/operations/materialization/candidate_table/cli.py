"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/candidate_table/cli.py

CLI for Eco1 candidate-table materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.candidate_table.pipeline import (
    materialize_candidate_table,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize Eco1 ProteinMPNN candidate_table.parquet.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = materialize_candidate_table(repo_root=args.repo_root, output_root=args.output_root)
    print(result.candidate_table_path)
    return 0
