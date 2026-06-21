"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/roster_cache/cli.py

CLI for Eco1 conservation roster source-cache materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.paths import (
    DEFAULT_CREATED_AT,
    DEFAULT_SOURCE_CACHE_ROOT,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.roster_cache.pipeline import (
    materialize_conservation_roster_cache,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the roster source-cache materialization CLI parser."""

    parser = argparse.ArgumentParser(description="Materialize Eco1 conservation roster source cache.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--roster-table", type=Path, required=True)
    parser.add_argument("--provider-source-root", type=Path, required=True)
    parser.add_argument("--provider-failure-ledger", type=Path)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_SOURCE_CACHE_ROOT)
    parser.add_argument("--created-at", default=DEFAULT_CREATED_AT)
    parser.add_argument(
        "--allow-uncontracted-roster-hash",
        action="store_true",
        help="Allow roster table hashes that do not match conservation-sources.yaml; intended for tests only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run roster source-cache materialization and print emitted paths."""

    args = build_parser().parse_args(argv)
    result = materialize_conservation_roster_cache(
        repo_root=args.repo_root,
        roster_table=args.roster_table,
        provider_source_root=args.provider_source_root,
        cache_root=args.cache_root,
        created_at=args.created_at,
        require_roster_source_hash=not args.allow_uncontracted_roster_hash,
        provider_failure_ledger=args.provider_failure_ledger,
    )
    print(
        json.dumps(
            {
                "cache_root": str(result.cache_root),
                "manifest_path": str(result.manifest_path),
                "provider_cache_paths": {key: str(value) for key, value in result.provider_cache_paths.items()},
                "source_records_path": str(result.source_records_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
