"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/promoter_candidate_bindings/cli.py

CLI for promoter candidate-binding previews and bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .artifact import materialize_promoter_candidate_bindings, verify_promoter_candidate_bindings
from .contracts import SCHEMA_ID, SCHEMA_VERSION, STUDY_ID
from .sources import preview_promoter_candidate_bindings_from_repo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage exact stress-study promoter candidate bindings.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preview = subparsers.add_parser("preview", help="Validate and summarize the study binding sources.")
    preview.add_argument("--repo-root", type=Path, required=True)
    materialize = subparsers.add_parser("materialize", help="Write manifest.json and bindings.parquet.")
    materialize.add_argument("--repo-root", type=Path, required=True)
    materialize.add_argument("--out-dir", type=Path)
    materialize.add_argument("--allowed-output-root", type=Path)
    materialize.add_argument("--overwrite", action="store_true")
    verify = subparsers.add_parser("verify", help="Verify a materialized binding bundle.")
    verify.add_argument("--bundle-dir", type=Path, required=True)
    verify.add_argument("--allowed-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preview":
        preview = preview_promoter_candidate_bindings_from_repo(args.repo_root)
        frame = preview.bindings
        _print(
            {
                "schema_id": SCHEMA_ID,
                "schema_version": SCHEMA_VERSION,
                "study_id": STUDY_ID,
                "binding_count": len(frame),
                "candidate_count": frame["candidate_id"].nunique(),
                "namespace_counts": frame["alias_namespace"].value_counts().sort_index().to_dict(),
                "adapter_counts": frame["baserender_adapter_kind"].value_counts().sort_index().to_dict(),
            }
        )
        return 0
    if args.command == "materialize":
        repo_root = args.repo_root.expanduser().resolve()
        default_root = (
            repo_root
            / "src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs"
            / "promoter_candidate_bindings"
        )
        allowed_root = (args.allowed_output_root or default_root).expanduser().resolve()
        out_dir = (args.out_dir or (allowed_root / "latest")).expanduser().resolve()
        result = materialize_promoter_candidate_bindings(
            preview_promoter_candidate_bindings_from_repo(repo_root),
            out_dir=out_dir,
            allowed_output_root=allowed_root,
            overwrite=bool(args.overwrite),
        )
        _print(
            {
                "manifest_json": str(result.manifest_json),
                "bindings_parquet": str(result.bindings_parquet),
                "binding_count": result.binding_count,
                "candidate_count": result.candidate_count,
            }
        )
        return 0
    verification = verify_promoter_candidate_bindings(args.bundle_dir, allowed_root=args.allowed_root)
    _print(
        {
            "schema_id": verification.schema_id,
            "schema_version": verification.schema_version,
            "study_id": verification.study_id,
            "binding_count": verification.binding_count,
            "candidate_count": verification.candidate_count,
        }
    )
    return 0


def _print(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


__all__ = ["build_parser", "main"]
