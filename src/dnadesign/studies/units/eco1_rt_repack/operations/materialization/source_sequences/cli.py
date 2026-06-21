"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/cli.py

CLI for Eco1 conservation source-sequence bundle materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.paths import (
    DEFAULT_CREATED_AT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_BUNDLE_ROOT,
    DEFAULT_SOURCE_CACHE_ROOT,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.pipeline import (
    materialize_source_sequence_bundles,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the source-sequence bundle materialization CLI parser."""

    parser = argparse.ArgumentParser(description="Materialize Eco1 RT conservation source-sequence bundles.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-cache-root", type=Path, default=DEFAULT_SOURCE_CACHE_ROOT)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_SOURCE_BUNDLE_ROOT)
    parser.add_argument("--created-at", default=DEFAULT_CREATED_AT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run source-sequence bundle materialization and print emitted paths."""

    args = build_parser().parse_args(argv)
    result = materialize_source_sequence_bundles(
        repo_root=args.repo_root,
        output_root=args.output_root,
        source_cache_root=args.source_cache_root,
        bundle_root=args.bundle_root,
        created_at=args.created_at,
    )
    print(
        json.dumps(
            {
                "bundle_manifest_path": str(result.bundle_manifest_path),
                "fasta_paths": {key: str(value) for key, value in result.fasta_paths.items()},
                "manifest_paths": {key: str(value) for key, value in result.manifest_paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
