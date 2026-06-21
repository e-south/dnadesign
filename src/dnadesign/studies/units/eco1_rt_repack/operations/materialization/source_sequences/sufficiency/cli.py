"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/sufficiency/cli.py

CLI for Eco1 source-sequence bundle sufficiency validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.paths import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_BUNDLE_ROOT,
    DEFAULT_SOURCE_CACHE_ROOT,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.sufficiency.pipeline import (
    validate_source_sequence_bundle_sufficiency,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Eco1 RT conservation source-sequence bundle sufficiency.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-cache-root", type=Path, default=DEFAULT_SOURCE_CACHE_ROOT)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_SOURCE_BUNDLE_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_source_sequence_bundle_sufficiency(
        repo_root=args.repo_root,
        output_root=args.output_root,
        source_cache_root=args.source_cache_root,
        bundle_root=args.bundle_root,
    )
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1
