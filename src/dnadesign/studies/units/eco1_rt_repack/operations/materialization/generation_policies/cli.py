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

from .candidate_pool import materialize_generation_policy_candidate_pool
from .foldcheck import materialize_generation_policy_foldcheck_request
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
    subparsers.add_parser("candidate-pool", help="Aggregate completed per-policy candidate tables.")
    subparsers.add_parser("foldcheck-request", help="Write a v2 ColabFold-ready FASTA and fold-check manifest.")
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
    if args.command == "candidate-pool":
        pool_result = materialize_generation_policy_candidate_pool(
            repo_root=args.repo_root,
            generation_policy_root=args.output_root,
        )
        print(pool_result.candidate_pool_path)
        print(pool_result.manifest_path)
    if args.command == "foldcheck-request":
        foldcheck_result = materialize_generation_policy_foldcheck_request(
            repo_root=args.repo_root,
            generation_policy_root=args.output_root,
            source_output_root=args.source_output_root,
        )
        print(foldcheck_result.input_fasta_path)
        print(foldcheck_result.request_manifest_path)
    return 0
