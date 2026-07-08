"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/generation_policies/cli.py

CLI for Eco1 RT v2 generation-policy materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import materialize_generation_policies
from .request_materialization import materialize_generation_policy_requests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize Eco1 RT v2 generation-policy manifests and ProteinMPNN request sidecars."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--source-output-root", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("policies", help="Write generation policy manifests only.")
    subparsers.add_parser("requests", help="Write per-policy ProteinMPNN request sidecars from an existing manifest.")
    subparsers.add_parser("all", help="Write policy manifests and per-policy ProteinMPNN request sidecars.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"policies", "all"}:
        policy_result = materialize_generation_policies(
            repo_root=args.repo_root,
            output_root=args.output_root,
            source_output_root=args.source_output_root,
        )
        print(policy_result.manifest_path)
    if args.command in {"requests", "all"}:
        request_result = materialize_generation_policy_requests(
            repo_root=args.repo_root,
            generation_policy_root=args.output_root,
            source_output_root=args.source_output_root,
        )
        for path in request_result.request_manifest_paths:
            print(path)
    return 0
