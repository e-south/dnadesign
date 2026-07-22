"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/proteinmpnn_sample_ingest/cli.py

CLI for Eco1 ProteinMPNN sample ingest.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.proteinmpnn_sample_ingest.pipeline import (
    materialize_proteinmpnn_samples,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ProteinMPNN and materialize Eco1 sample_table.parquet.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--proteinmpnn-root", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output batch with the same batch_id.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = materialize_proteinmpnn_samples(
        repo_root=args.repo_root,
        output_root=args.output_root,
        proteinmpnn_root=args.proteinmpnn_root,
        overwrite=args.overwrite,
    )
    print(result.sample_table_path)
    return 0
