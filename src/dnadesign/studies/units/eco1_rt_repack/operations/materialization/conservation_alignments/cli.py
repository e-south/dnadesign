"""CLI for Eco1 conservation alignment bundle materialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.conservation_alignments.pipeline import (
    DEFAULT_ALIGNMENT_BUNDLE_ROOT,
    DEFAULT_CREATED_AT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_BUNDLE_ROOT,
    DEFAULT_SOURCE_CACHE_ROOT,
    materialize_conservation_alignment_bundles,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the conservation-alignment bundle materialization CLI parser."""

    parser = argparse.ArgumentParser(description="Materialize Eco1 RT conservation alignment bundles.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-cache-root", type=Path, default=DEFAULT_SOURCE_CACHE_ROOT)
    parser.add_argument("--source-bundle-root", type=Path, default=DEFAULT_SOURCE_BUNDLE_ROOT)
    parser.add_argument("--alignment-root", type=Path, default=DEFAULT_ALIGNMENT_BUNDLE_ROOT)
    parser.add_argument(
        "--profile-id",
        action="append",
        dest="profile_ids",
        help="Run only the selected declared conservation profile id. Repeat for multiple profiles.",
    )
    parser.add_argument("--created-at", default=DEFAULT_CREATED_AT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run conservation-alignment bundle materialization and print emitted paths."""

    args = build_parser().parse_args(argv)
    result = materialize_conservation_alignment_bundles(
        repo_root=args.repo_root,
        output_root=args.output_root,
        source_cache_root=args.source_cache_root,
        source_bundle_root=args.source_bundle_root,
        alignment_root=args.alignment_root,
        profile_ids=tuple(args.profile_ids) if args.profile_ids else None,
        created_at=args.created_at,
    )
    print(
        json.dumps(
            {
                "aligned_fasta_paths": {key: str(value) for key, value in result.aligned_fasta_paths.items()},
                "bundle_manifest_path": str(result.bundle_manifest_path),
                "manifest_paths": {key: str(value) for key, value in result.manifest_paths.items()},
                "total_elapsed_seconds": result.total_elapsed_seconds,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
