"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/design_classes/cli.py

CLI for Eco1 RT design-class expansion materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes import (
    materialize_design_class_candidate_pool,
    materialize_design_class_foldcheck_request,
    materialize_design_class_requests,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.constants import (
    DEFAULT_CREATED_AT,
    DEFAULT_DESIGN_CLASSES_ROOT,
    DEFAULT_SOURCE_OUTPUT_ROOT,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the design-class materialization CLI parser."""

    parser = argparse.ArgumentParser(description="Materialize Eco1 RT design-class expansion artifacts.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DESIGN_CLASSES_ROOT)
    parser.add_argument("--source-output-root", type=Path, default=DEFAULT_SOURCE_OUTPUT_ROOT)
    parser.add_argument("--created-at", default=DEFAULT_CREATED_AT)
    subparsers = parser.add_subparsers(dest="command")
    requests = subparsers.add_parser("requests", help="Write mask sets, thread plans, and ProteinMPNN requests.")
    requests.add_argument("--class-id", action="append", default=None)
    pool = subparsers.add_parser("candidate-pool", help="Write nonredundant candidate pool from available tables.")
    pool.add_argument("--baseline-candidate-table-path", type=Path, default=None)
    subparsers.add_parser("foldcheck-request", help="Write ColabFold request for the nonredundant candidate pool.")
    subparsers.add_parser("all-local", help="Run local non-execution materialization steps.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run design-class materialization and print emitted paths."""

    args = build_parser().parse_args(argv)
    command = args.command or "requests"
    payload: dict[str, object] = {}
    try:
        if command in {"requests", "all-local"}:
            result = materialize_design_class_requests(
                repo_root=args.repo_root,
                output_root=args.output_root,
                source_output_root=args.source_output_root,
                class_ids=getattr(args, "class_id", None),
                created_at=args.created_at,
            )
            payload["design_class_manifest_path"] = str(result.manifest_path)
            payload["class_request_manifest_paths"] = [
                str(artifact.request_manifest_path) for artifact in result.class_artifacts
            ]
        if command in {"candidate-pool", "all-local"}:
            result = materialize_design_class_candidate_pool(
                repo_root=args.repo_root,
                output_root=args.output_root,
                source_output_root=args.source_output_root,
                baseline_candidate_table_path=getattr(args, "baseline_candidate_table_path", None),
            )
            payload["candidate_pool_path"] = str(result.candidate_pool_path)
            payload["candidate_pool_manifest_path"] = str(result.manifest_path)
        if command in {"foldcheck-request", "all-local"}:
            result = materialize_design_class_foldcheck_request(
                repo_root=args.repo_root,
                output_root=args.output_root,
                source_output_root=args.source_output_root,
                created_at=args.created_at,
            )
            payload["foldcheck_input_fasta_path"] = str(result.input_fasta_path)
            payload["foldcheck_request_manifest_path"] = str(result.request_manifest_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
