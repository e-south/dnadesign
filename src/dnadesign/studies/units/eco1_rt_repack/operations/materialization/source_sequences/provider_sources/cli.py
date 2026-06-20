"""CLI for Eco1 provider FASTA source acquisition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.paths import (
    DEFAULT_CREATED_AT,
    DEFAULT_PROVIDER_SOURCE_ROOT,
)

from .pipeline import materialize_provider_source_fastas


def build_parser() -> argparse.ArgumentParser:
    """Build the provider FASTA source acquisition CLI parser."""

    parser = argparse.ArgumentParser(description="Materialize Eco1 provider FASTA source files.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--roster-table", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_PROVIDER_SOURCE_ROOT)
    parser.add_argument("--created-at", default=DEFAULT_CREATED_AT)
    parser.add_argument(
        "--write-unresolved-ledger",
        action="store_true",
        help="Write explicit failure ledger for provider-missing accessions instead of failing immediately.",
    )
    parser.add_argument(
        "--allow-uncontracted-roster-hash",
        action="store_true",
        help="Allow roster table hashes that do not match conservation-sources.yaml; intended for tests only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run provider FASTA source acquisition and print emitted paths."""

    args = build_parser().parse_args(argv)
    result = materialize_provider_source_fastas(
        repo_root=args.repo_root,
        roster_table=args.roster_table,
        source_root=args.source_root,
        created_at=args.created_at,
        require_roster_source_hash=not args.allow_uncontracted_roster_hash,
        write_unresolved_ledger=args.write_unresolved_ledger,
    )
    print(
        json.dumps(
            {
                "source_root": str(result.source_root),
                "manifest_path": str(result.manifest_path),
                "failure_ledger_path": str(result.failure_ledger_path) if result.failure_ledger_path else None,
                "fasta_paths": {key: str(value) for key, value in result.fasta_paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
