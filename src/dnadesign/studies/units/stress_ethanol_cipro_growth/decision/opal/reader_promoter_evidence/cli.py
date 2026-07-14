"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/reader_promoter_evidence/cli.py

CLI for study-owned Reader promoter-evidence display manifests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .contracts import READER_EVIDENCE_SCHEMA_VERSION, READER_PROMOTER_EVIDENCE_FILENAME
from .manifest import (
    materialize_reader_promoter_evidence_manifest,
    preview_reader_promoter_evidence_manifest,
    verify_reader_promoter_evidence_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage verified Reader promoter evidence for static OPAL display.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview = subparsers.add_parser("preview", help="Validate Reader bundles and preview the display manifest.")
    _add_bundle_inputs(preview)

    materialize = subparsers.add_parser("materialize", help="Atomically write the display manifest.")
    _add_bundle_inputs(materialize)
    materialize.add_argument("--out-dir", type=Path, required=True)
    materialize.add_argument("--filename", default=READER_PROMOTER_EVIDENCE_FILENAME)
    materialize.add_argument("--overwrite", action="store_true")

    verify = subparsers.add_parser("verify", help="Verify a materialized display manifest and its Reader sources.")
    verify.add_argument("manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preview":
        payload = preview_reader_promoter_evidence_manifest(
            args.bundle_dirs,
            bindings_bundle=args.bindings_bundle,
            round_label=args.round_label,
        )
    elif args.command == "materialize":
        result = materialize_reader_promoter_evidence_manifest(
            args.bundle_dirs,
            bindings_bundle=args.bindings_bundle,
            out_dir=args.out_dir,
            round_label=args.round_label,
            filename=args.filename,
            overwrite=bool(args.overwrite),
        )
        payload = {
            "schema_version": READER_EVIDENCE_SCHEMA_VERSION,
            "manifest_json": str(result.manifest_json),
            "row_count": result.row_count,
            "artifact_count": result.artifact_count,
        }
    else:
        verification = verify_reader_promoter_evidence_manifest(args.manifest)
        payload = {
            "schema_version": READER_EVIDENCE_SCHEMA_VERSION,
            "manifest_json": str(verification.manifest_json),
            "row_count": verification.row_count,
            "artifact_count": verification.artifact_count,
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _add_bundle_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--round", dest="round_label", default="r0")
    parser.add_argument(
        "--bindings-bundle",
        type=Path,
        required=True,
        help="Verified study-owned promoter-candidate binding bundle.",
    )
    parser.add_argument("bundle_dirs", type=Path, nargs="+")


__all__ = ["build_parser", "main"]
